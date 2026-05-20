//! 4-bit Product Quantization with SIMD shuffle ADC ("FastScan").
//!
//! Layout: vectors are grouped in blocks of 32. For each sub-quantizer
//! `s ∈ [0, m)` the block stores 16 bytes. Byte `b` packs two codes:
//!   * low nibble  → vector `b`        within the block (lane 0..15)
//!   * high nibble → vector `b + 16`   within the block (lane 16..31)
//!
//! Scan kernel per block:
//!   1. Per-query 4-bit LUT, `m * 16` u8 entries (quantized from f32).
//!   2. For each sub-quantizer load 16 code bytes; mask + shift into the
//!      two index halves; do two `vqtbl1q_u8` table lookups.
//!   3. Zero-extend the resulting `u8x16` lanes into `u16x8` accumulators.
//!
//! `vqtbl1q_u8` on AArch64 is a single-cycle 16-byte gather; the same
//! kernel maps onto x86 AVX2 `_mm256_shuffle_epi8` with the standard
//! per-lane workaround. Only NEON is implemented here — the scalar
//! fallback below produces bit-identical accumulator output.

use crate::kmeans;
use crate::pq::ProductQuantizer;

/// Vectors per FastScan storage block.
pub const BLOCK: usize = 32;

/// LUT entries per sub-quantizer (4-bit codes ⇒ 16).
pub const K_FS: usize = 16;

/// Per-query u8-quantized distance lookup table.
///
/// `entries[s * K_FS + c]` is the u8-scaled squared L2 between the query
/// subvector `s` and centroid `c` of subspace `s`. The scale is stored on
/// the LUT so a u16 accumulator sum can be back-projected to f32 if needed.
pub struct FastScanLut {
    pub m: usize,
    pub entries: Vec<u8>,
    pub scale: f32, // approx_f32 = (u16_sum as f32) * scale
}

/// FastScan index. Database codes are stored block-by-block, sub-quantizer
/// inner. Total size: `ceil(n / BLOCK) * m * 16` bytes.
pub struct FastScanIndex {
    pub pq: ProductQuantizer, // pq.k == K_FS, pq.m == m
    pub n: usize,
    pub blocks: usize, // ceil(n / BLOCK)
    pub packed: Vec<u8>, // blocks * m * 16
}

impl FastScanIndex {
    /// Train K=16 PQ and pack `n` vectors into the FastScan layout.
    pub fn from_vectors(
        train: &[f32],
        n_train: usize,
        vectors: &[f32],
        n: usize,
        dim: usize,
        m: usize,
        iters: usize,
        seed: u64,
    ) -> crate::Result<Self> {
        let pq = ProductQuantizer::train(train, n_train, dim, m, K_FS, iters, seed)?;
        let codes = pq.encode(vectors, n); // n * m, each u8 in 0..16
        let blocks = (n + BLOCK - 1) / BLOCK;
        let mut packed = vec![0u8; blocks * m * 16];

        for b in 0..blocks {
            let v_lo_base = b * BLOCK;
            for s in 0..m {
                let chunk = &mut packed[b * m * 16 + s * 16..b * m * 16 + (s + 1) * 16];
                for i in 0..16 {
                    let v_lo = v_lo_base + i;
                    let v_hi = v_lo_base + i + 16;
                    let lo = if v_lo < n { codes[v_lo * m + s] & 0x0F } else { 0 };
                    let hi = if v_hi < n { codes[v_hi * m + s] & 0x0F } else { 0 };
                    chunk[i] = (hi << 4) | lo;
                }
            }
        }

        Ok(Self { pq, n, blocks, packed })
    }

    /// Build the per-query u8 LUT.
    ///
    /// Per-sub-quantizer minimum is subtracted (a constant across all DB
    /// vectors — does not affect ranking). The residual is then globally
    /// scaled so the worst-case row residual maps to 255. This is the same
    /// quantization scheme used by FAISS / ScaNN; without the per-row
    /// min-subtraction, a single outlier centroid in one sub-quantizer
    /// collapses precision on every other sub-quantizer (recall collapses
    /// to near-random).
    pub fn build_lut(&self, query: &[f32]) -> FastScanLut {
        let f = self.pq.build_lut_f32(query);
        let m = self.pq.m;
        let kc = K_FS;
        let mut row_min = vec![0f32; m];
        let mut row_range = vec![0f32; m];
        for s in 0..m {
            let row = &f[s * kc..(s + 1) * kc];
            let mn = row.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            row_min[s] = mn;
            row_range[s] = (mx - mn).max(1e-12);
        }
        let max_range = row_range.iter().copied().fold(0f32, f32::max).max(1e-12);
        let scale_q = 255.0 / max_range;
        let mut entries = vec![0u8; m * kc];
        for s in 0..m {
            for c in 0..kc {
                let v = f[s * kc + c] - row_min[s];
                entries[s * kc + c] = (v * scale_q).round().clamp(0.0, 255.0) as u8;
            }
        }
        // Approx f32 distance ≈ (u16_sum / scale_q) + sum(row_min).
        // We store only the inverse scale; the constant offset cancels in
        // any ranking-only consumer and the demo prints raw sums.
        FastScanLut { m, entries, scale: 1.0 / scale_q }
    }

    /// Scan all blocks. Returns top-`k` (vector_idx, u16_sum) ascending.
    /// Sum is the raw u16 accumulator; multiply by `lut.scale` for an
    /// approximate squared-L2 estimate.
    pub fn search_u16(&self, lut: &FastScanLut, k: usize) -> Vec<(u32, u16)> {
        assert_eq!(lut.m, self.pq.m);
        let m = self.pq.m;
        let mut sums = vec![0u16; self.blocks * BLOCK];

        for b in 0..self.blocks {
            let block_codes = &self.packed[b * m * 16..(b + 1) * m * 16];
            let block_sums = &mut sums[b * BLOCK..(b + 1) * BLOCK];
            scan_block(block_codes, &lut.entries, m, block_sums);
        }

        // Truncate padding past n, then partial sort for top-k.
        sums.truncate(self.n);
        let mut idx: Vec<(u32, u16)> = sums.into_iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        idx.sort_unstable_by_key(|&(_, s)| s);
        idx.truncate(k);
        idx
    }

    /// Approximate squared-L2 estimate for raw block-sum `s` under `lut`.
    pub fn approx_sq_l2(&self, s: u16, lut: &FastScanLut) -> f32 {
        s as f32 * lut.scale
    }

    /// Storage size in bytes for the packed codes (excluding the codebook).
    pub fn packed_bytes(&self) -> usize {
        self.packed.len()
    }

    /// Two-stage retrieval: FastScan returns top-`candidates`, then those
    /// candidates are reranked against full-precision `db_vectors`.
    /// Returns top-`k` (idx, exact_sq_l2).
    ///
    /// This is the standard production setup. The PQ scan stage is the
    /// filter (fast, lossy, u8-quantized); the rerank stage is the
    /// arbiter (slow, exact, f32). For `candidates >> k` recall@k
    /// approaches 1.0 while throughput stays close to scan-only.
    pub fn search_rerank(
        &self,
        lut: &FastScanLut,
        query: &[f32],
        db_vectors: &[f32],
        dim: usize,
        candidates: usize,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let cands = self.search_u16(lut, candidates);
        let mut rer: Vec<(u32, f32)> = cands.iter()
            .map(|&(i, _)| {
                let v = &db_vectors[i as usize * dim..(i as usize + 1) * dim];
                (i, crate::sq_l2(v, query))
            })
            .collect();
        rer.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rer.truncate(k);
        rer
    }
}

/// Scan a single 32-vector block. Dispatches to NEON when available.
#[inline]
pub fn scan_block(block_codes: &[u8], lut: &[u8], m: usize, out: &mut [u16]) {
    debug_assert_eq!(block_codes.len(), m * 16);
    debug_assert_eq!(out.len(), BLOCK);
    #[cfg(target_arch = "aarch64")]
    unsafe { return scan_block_neon(block_codes, lut, m, out); }
    #[allow(unreachable_code)]
    scan_block_scalar(block_codes, lut, m, out);
}

/// Reference scalar kernel — must produce bit-identical output to NEON.
pub fn scan_block_scalar(block_codes: &[u8], lut: &[u8], m: usize, out: &mut [u16]) {
    for v in out.iter_mut() { *v = 0; }
    for s in 0..m {
        let chunk = &block_codes[s * 16..(s + 1) * 16];
        let row = &lut[s * K_FS..(s + 1) * K_FS];
        for i in 0..16 {
            let lo = (chunk[i] & 0x0F) as usize;
            let hi = (chunk[i] >> 4) as usize;
            out[i]      = out[i].wrapping_add(row[lo] as u16);
            out[i + 16] = out[i + 16].wrapping_add(row[hi] as u16);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scan_block_neon(block_codes: &[u8], lut: &[u8], m: usize, out: &mut [u16]) {
    use core::arch::aarch64::*;
    // Four u16x8 lane groups: lanes 0..7, 8..15, 16..23, 24..31.
    let mut acc0 = vdupq_n_u16(0);
    let mut acc1 = vdupq_n_u16(0);
    let mut acc2 = vdupq_n_u16(0);
    let mut acc3 = vdupq_n_u16(0);
    let mask_lo = vdupq_n_u8(0x0F);

    for s in 0..m {
        let codes = vld1q_u8(block_codes.as_ptr().add(s * 16));
        let row = vld1q_u8(lut.as_ptr().add(s * K_FS));
        let idx_lo = vandq_u8(codes, mask_lo);
        let idx_hi = vshrq_n_u8(codes, 4);
        let d_lo = vqtbl1q_u8(row, idx_lo);
        let d_hi = vqtbl1q_u8(row, idx_hi);
        // Widen and accumulate.
        acc0 = vaddq_u16(acc0, vmovl_u8(vget_low_u8(d_lo)));
        acc1 = vaddq_u16(acc1, vmovl_u8(vget_high_u8(d_lo)));
        acc2 = vaddq_u16(acc2, vmovl_u8(vget_low_u8(d_hi)));
        acc3 = vaddq_u16(acc3, vmovl_u8(vget_high_u8(d_hi)));
    }

    vst1q_u16(out.as_mut_ptr(), acc0);
    vst1q_u16(out.as_mut_ptr().add(8), acc1);
    vst1q_u16(out.as_mut_ptr().add(16), acc2);
    vst1q_u16(out.as_mut_ptr().add(24), acc3);
}

#[allow(dead_code)]
fn _train_codebook_for_test(train: &[f32], n: usize, d: usize) -> Vec<f32> {
    // 16 centroids for a single subspace — used by unit tests via `pub(crate)`.
    kmeans::kmeans(train, n, d, K_FS, 8, 0xC0FFEE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_block_codes(m: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..m * 16).map(|_| rng.gen::<u8>()).collect()
    }

    fn random_lut(m: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..m * K_FS).map(|_| rng.gen::<u8>()).collect()
    }

    #[test]
    fn neon_matches_scalar_small_m() {
        let m = 8usize;
        let codes = random_block_codes(m, 1);
        let lut = random_lut(m, 2);
        let mut a = vec![0u16; BLOCK];
        let mut b = vec![0u16; BLOCK];
        scan_block_scalar(&codes, &lut, m, &mut a);
        scan_block(&codes, &lut, m, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn neon_matches_scalar_large_m() {
        let m = 32usize;
        let codes = random_block_codes(m, 3);
        let lut = random_lut(m, 4);
        let mut a = vec![0u16; BLOCK];
        let mut b = vec![0u16; BLOCK];
        scan_block_scalar(&codes, &lut, m, &mut a);
        scan_block(&codes, &lut, m, &mut b);
        assert_eq!(a, b);
    }
}
