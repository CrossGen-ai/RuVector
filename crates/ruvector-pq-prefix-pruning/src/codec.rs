//! 4-bit PQ codec: M segments, N codes, with FastScan-style transposed layout.
//!
//! Canonical layout: for each segment m (0..M), the N nibble codes are packed
//! contiguously (two codes per byte, low nibble = even id). This is the layout
//! FastScan uses so that per-segment LUT gathers are sequential in memory.
//!
//! We store codes transposed as `data[m * ceil(N/2) + (id/2)]`. The group-of-16
//! micro-layout used by SIMD LUT16 is handled by the scanner, not the codec,
//! so the codec stays portable.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct PqParams {
    /// Dimensions per segment.
    pub d_sub: usize,
    /// Number of segments.
    pub m: usize,
    /// Codebook size — fixed at 16 for 4-bit codes.
    pub k_star: usize,
}

impl PqParams {
    pub fn new(d: usize, m: usize) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        Self { d_sub: d / m, m, k_star: 16 }
    }
    pub fn d(&self) -> usize {
        self.d_sub * self.m
    }
}

/// A PQ codebook: per-segment set of 16 centroids of dimension d_sub.
#[derive(Clone, Debug)]
pub struct Codebook {
    pub params: PqParams,
    /// Flat: m * 16 * d_sub.
    pub centroids: Vec<f32>,
}

impl Codebook {
    pub fn centroid(&self, m: usize, c: usize) -> &[f32] {
        let d = self.params.d_sub;
        let off = (m * 16 + c) * d;
        &self.centroids[off..off + d]
    }

    /// Train a codebook on `vectors` (count × d) using mini-batch k-means
    /// (lloyd, cosine-of-no-op simple L2). Deterministic given `seed`.
    pub fn train(vectors: &[f32], count: usize, params: PqParams, iters: usize, seed: u64) -> Self {
        let d = params.d();
        assert_eq!(vectors.len(), count * d);
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut centroids = vec![0.0f32; params.m * 16 * params.d_sub];

        for m in 0..params.m {
            // Pick 16 distinct random starts.
            let mut picks: Vec<usize> = (0..count).collect();
            for i in 0..16 {
                let j = i + rng.gen_range(0..(count - i));
                picks.swap(i, j);
            }
            for c in 0..16 {
                let src = picks[c];
                let seg_off = m * params.d_sub;
                let dst = (m * 16 + c) * params.d_sub;
                centroids[dst..dst + params.d_sub]
                    .copy_from_slice(&vectors[src * d + seg_off..src * d + seg_off + params.d_sub]);
            }

            // Lloyd iterations on this segment.
            let mut counts = [0u32; 16];
            let mut sums = vec![0.0f32; 16 * params.d_sub];
            for _ in 0..iters {
                counts.fill(0);
                sums.fill(0.0);
                for i in 0..count {
                    let x = &vectors[i * d + m * params.d_sub..i * d + (m + 1) * params.d_sub];
                    let mut best = 0usize;
                    let mut bd = f32::INFINITY;
                    for c in 0..16 {
                        let cent = &centroids[(m * 16 + c) * params.d_sub
                            ..(m * 16 + c + 1) * params.d_sub];
                        let dsq = l2sq(x, cent);
                        if dsq < bd {
                            bd = dsq;
                            best = c;
                        }
                    }
                    counts[best] += 1;
                    for (s, v) in sums[best * params.d_sub..(best + 1) * params.d_sub]
                        .iter_mut()
                        .zip(x.iter())
                    {
                        *s += *v;
                    }
                }
                for c in 0..16 {
                    if counts[c] > 0 {
                        let inv = 1.0 / counts[c] as f32;
                        for (c_out, s) in centroids
                            [(m * 16 + c) * params.d_sub..(m * 16 + c + 1) * params.d_sub]
                            .iter_mut()
                            .zip(sums[c * params.d_sub..(c + 1) * params.d_sub].iter())
                        {
                            *c_out = s * inv;
                        }
                    }
                }
            }
        }

        Self { params, centroids }
    }
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// 4-bit codes in a FastScan-ready transposed layout.
///
/// `data[m * nbytes_per_seg + (id / 2)]` holds code[m][id] in its low nibble
/// when id is even, high nibble when id is odd.
#[derive(Clone, Debug)]
pub struct PqCodes {
    pub n: usize,
    pub params: PqParams,
    /// Transposed storage: m rows of ceil(N/2) bytes.
    pub data: Vec<u8>,
    pub bytes_per_seg: usize,
}

impl PqCodes {
    pub fn encode(vectors: &[f32], count: usize, book: &Codebook) -> Self {
        let d = book.params.d();
        let m = book.params.m;
        let bytes_per_seg = (count + 1) / 2;
        let mut data = vec![0u8; m * bytes_per_seg];
        for seg in 0..m {
            for id in 0..count {
                let x = &vectors[id * d + seg * book.params.d_sub
                    ..id * d + (seg + 1) * book.params.d_sub];
                let mut best = 0u8;
                let mut bd = f32::INFINITY;
                for c in 0..16u8 {
                    let cent = book.centroid(seg, c as usize);
                    let dsq = l2sq(x, cent);
                    if dsq < bd {
                        bd = dsq;
                        best = c;
                    }
                }
                let b = &mut data[seg * bytes_per_seg + id / 2];
                if id % 2 == 0 {
                    *b = (*b & 0xF0) | (best & 0x0F);
                } else {
                    *b = (*b & 0x0F) | ((best & 0x0F) << 4);
                }
            }
        }
        Self { n: count, params: book.params.clone(), data, bytes_per_seg }
    }

    #[inline]
    pub fn code(&self, seg: usize, id: usize) -> u8 {
        let b = self.data[seg * self.bytes_per_seg + id / 2];
        if id % 2 == 0 { b & 0x0F } else { b >> 4 }
    }
}
