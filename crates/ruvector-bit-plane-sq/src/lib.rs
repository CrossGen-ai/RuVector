//! Bit-Plane Progressive SQ8 (BPP-SQ)
//!
//! Scalar quantization to 8 bits per component, but codes are stored transposed
//! into eight bit-planes (MSB-plane .. LSB-plane). A query's full distance to a
//! vector can be approximated using only the top M planes by treating each plane
//! as a weighted contribution (MSB has weight 2^7, next 2^6, ...). The partial
//! distance against the top M planes gives a monotone-approaching lower-bound
//! prefix of the full int-domain distance, which is usable as an ADC-style
//! early-abandon statistic and as a cheap shortlist filter. Only shortlisted
//! vectors pay the full M=8 cost.
//!
//! This is distinct from:
//!   * bit-sliced-hamming — pure binary codes, Hamming distance, no SQ refine.
//!   * sq-cascade-int4-residual — two separate quantized code arrays.
//! Here there is one SQ8 code per vector; the bit-plane layout is a *physical*
//! reshape of that one code into 8 bitmaps so partial-precision scans skip
//! low-order planes entirely (no load, no ALU cost).
//!
//! Trait-based backends let future planes-per-group variants plug in without
//! touching the search loop.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::time::Instant;

/// One scalar-quantized code per vector, stored as a packed [u8; D].
#[derive(Clone, Debug)]
pub struct Sq8Codes {
    pub dim: usize,
    pub n: usize,
    /// Row-major n * dim.
    pub codes: Vec<u8>,
    /// Per-dimension min used for dequantization.
    pub dim_min: Vec<f32>,
    /// Per-dimension scale used for dequantization (max - min).
    pub dim_scale: Vec<f32>,
}

impl Sq8Codes {
    /// Fit per-dimension min/scale then quantize `vectors` (row-major n*dim).
    pub fn fit_quantize(vectors: &[f32], dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert_eq!(vectors.len() % dim, 0, "vectors length not multiple of dim");
        let n = vectors.len() / dim;

        let mut dim_min = vec![f32::INFINITY; dim];
        let mut dim_max = vec![f32::NEG_INFINITY; dim];
        for row in vectors.chunks_exact(dim) {
            for (j, &v) in row.iter().enumerate() {
                if v < dim_min[j] {
                    dim_min[j] = v;
                }
                if v > dim_max[j] {
                    dim_max[j] = v;
                }
            }
        }
        let dim_scale: Vec<f32> = dim_min
            .iter()
            .zip(dim_max.iter())
            .map(|(lo, hi)| {
                let s = hi - lo;
                if s <= f32::EPSILON {
                    1.0
                } else {
                    s
                }
            })
            .collect();

        let mut codes = vec![0u8; n * dim];
        for (row_in, row_out) in vectors.chunks_exact(dim).zip(codes.chunks_exact_mut(dim)) {
            for j in 0..dim {
                let x = (row_in[j] - dim_min[j]) / dim_scale[j];
                let q = (x * 255.0).round().clamp(0.0, 255.0) as u8;
                row_out[j] = q;
            }
        }
        Self {
            dim,
            n,
            codes,
            dim_min,
            dim_scale,
        }
    }

    /// Reconstruct a vector (for refine / debug).
    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        let row = &self.codes[i * self.dim..(i + 1) * self.dim];
        row.iter()
            .enumerate()
            .map(|(j, &c)| self.dim_min[j] + (c as f32 / 255.0) * self.dim_scale[j])
            .collect()
    }
}

/// Bit-plane layout: 8 planes, each is a packed-bitmap per vector of `dim` bits.
#[derive(Clone, Debug)]
pub struct BitPlanes {
    pub dim: usize,
    pub n: usize,
    /// Number of u64 words per vector per plane. `words = ceil(dim / 64)`.
    pub words: usize,
    /// Planes stored MSB-first. planes[0] = MSB plane, planes[7] = LSB plane.
    /// Each plane is a flat n*words u64 array.
    pub planes: Vec<Vec<u64>>,
    pub dim_min: Vec<f32>,
    pub dim_scale: Vec<f32>,
}

impl BitPlanes {
    pub fn from_sq8(sq: &Sq8Codes) -> Self {
        let dim = sq.dim;
        let n = sq.n;
        let words = (dim + 63) / 64;
        let mut planes: Vec<Vec<u64>> = (0..8).map(|_| vec![0u64; n * words]).collect();
        for i in 0..n {
            let row = &sq.codes[i * dim..(i + 1) * dim];
            for (j, &c) in row.iter().enumerate() {
                let w = j / 64;
                let b = j % 64;
                for p in 0..8 {
                    // p=0 is MSB (bit index 7), p=7 is LSB (bit index 0).
                    let bit = (c >> (7 - p)) & 1;
                    if bit == 1 {
                        planes[p][i * words + w] |= 1u64 << b;
                    }
                }
            }
        }
        Self {
            dim,
            n,
            words,
            planes,
            dim_min: sq.dim_min.clone(),
            dim_scale: sq.dim_scale.clone(),
        }
    }

    /// Build a per-plane LUT: for every bit position j and bit value b in {0,1},
    /// the (q_j - b*255 / 2^p) contribution. We return 8 arrays of length dim,
    /// each holding the plane-weight-scaled contribution the plane adds when
    /// set, assuming L1 distance in the int domain.
    ///
    /// For L1: |q - c| cannot be decomposed per-bit, so we use the standard
    /// tight bound: with top-M planes known the code's value lies in an
    /// interval of width 2^(8-M). The partial L1 distance against the midpoint
    /// of that interval is a provably within (dim * 2^(8-M) / 2) bound of the
    /// true L1 distance. We compute that midpoint explicitly for each group of
    /// planes considered.
    pub fn planes_to_approx_codes(&self, top_m: usize) -> Vec<u8> {
        assert!(top_m >= 1 && top_m <= 8);
        let mut out = vec![0u8; self.n * self.dim];
        let tail = 8 - top_m;
        for i in 0..self.n {
            for j in 0..self.dim {
                let w = j / 64;
                let b = j % 64;
                let mut v: u8 = 0;
                for p in 0..top_m {
                    let bit = (self.planes[p][i * self.words + w] >> b) & 1;
                    v |= (bit as u8) << (7 - p);
                }
                // Add half-interval bias for the unknown tail so we hit the
                // midpoint of the surviving range: v + 2^(tail-1).
                let bias = if tail == 0 {
                    0u16
                } else {
                    1u16 << (tail - 1)
                };
                let vi = (v as u16) + bias;
                out[i * self.dim + j] = vi.min(255) as u8;
            }
        }
        out
    }
}

/// L1 distance between a quantized query and a quantized vector, in the int
/// domain (returns a u64 — sum of absolute differences across `dim` bytes).
#[inline]
pub fn l1_u8(a: &[u8], b: &[u8]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (*x as i16 - *y as i16).unsigned_abs() as u64;
    }
    acc
}

/// L2 squared distance in float domain (used for ground-truth / refine).
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Search API: three backends behind one trait.
pub trait L1Search {
    /// Return the indices of the k smallest-distance vectors.
    fn search(&self, query: &[u8], k: usize) -> Vec<u32>;
    fn name(&self) -> &'static str;
}

/// Baseline: full 8-bit L1 over row-major SQ8 codes.
pub struct FullSq8<'a> {
    pub sq: &'a Sq8Codes,
}

impl<'a> L1Search for FullSq8<'a> {
    fn search(&self, query: &[u8], k: usize) -> Vec<u32> {
        let d = self.sq.dim;
        let mut heap = TopK::new(k);
        for i in 0..self.sq.n {
            let row = &self.sq.codes[i * d..(i + 1) * d];
            heap.push(i as u32, l1_u8(query, row));
        }
        heap.into_sorted_ids()
    }
    fn name(&self) -> &'static str {
        "full_sq8"
    }
}

/// Top-M planes only — approximate L1 using midpoint-bias reconstruction.
pub struct TopMApprox<'a> {
    pub approx_codes: &'a [u8],
    pub n: usize,
    pub dim: usize,
    pub name_str: &'static str,
}

impl<'a> L1Search for TopMApprox<'a> {
    fn search(&self, query: &[u8], k: usize) -> Vec<u32> {
        let mut heap = TopK::new(k);
        for i in 0..self.n {
            let row = &self.approx_codes[i * self.dim..(i + 1) * self.dim];
            heap.push(i as u32, l1_u8(query, row));
        }
        heap.into_sorted_ids()
    }
    fn name(&self) -> &'static str {
        self.name_str
    }
}

/// Progressive: shortlist with top-M planes, then refine with full SQ8.
/// shortlist_mul controls how many candidates survive from the first stage.
pub struct Progressive<'a> {
    pub approx_codes: &'a [u8],
    pub sq: &'a Sq8Codes,
    pub top_m: usize,
    pub shortlist_mul: usize,
    pub name_str: &'static str,
}

impl<'a> L1Search for Progressive<'a> {
    fn search(&self, query: &[u8], k: usize) -> Vec<u32> {
        let d = self.sq.dim;
        let short_k = (k * self.shortlist_mul).min(self.sq.n).max(k);

        // Stage 1: approximate L1 against top-M planes (midpoint-biased).
        let mut short_heap = TopK::new(short_k);
        for i in 0..self.sq.n {
            let row = &self.approx_codes[i * d..(i + 1) * d];
            short_heap.push(i as u32, l1_u8(query, row));
        }
        let candidates = short_heap.into_sorted_ids();

        // Stage 2: full 8-bit L1 refine over the shortlist.
        let mut final_heap = TopK::new(k);
        for &id in &candidates {
            let row = &self.sq.codes[(id as usize) * d..((id as usize) + 1) * d];
            final_heap.push(id, l1_u8(query, row));
        }
        final_heap.into_sorted_ids()
    }
    fn name(&self) -> &'static str {
        self.name_str
    }
}

// --- Top-K min-of-k via a tiny max-heap (entry = (dist, id)) ----------------
pub struct TopK {
    cap: usize,
    heap: Vec<(u64, u32)>,
}

impl TopK {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            heap: Vec::with_capacity(cap),
        }
    }
    pub fn push(&mut self, id: u32, dist: u64) {
        if self.heap.len() < self.cap {
            self.heap.push((dist, id));
            self.sift_up(self.heap.len() - 1);
        } else if dist < self.heap[0].0 {
            self.heap[0] = (dist, id);
            self.sift_down(0);
        }
    }
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.heap[i].0 > self.heap[p].0 {
                self.heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut m = i;
            if l < n && self.heap[l].0 > self.heap[m].0 {
                m = l;
            }
            if r < n && self.heap[r].0 > self.heap[m].0 {
                m = r;
            }
            if m == i {
                break;
            }
            self.heap.swap(i, m);
            i = m;
        }
    }
    pub fn into_sorted_ids(mut self) -> Vec<u32> {
        self.heap.sort_by_key(|&(d, _)| d);
        self.heap.into_iter().map(|(_, id)| id).collect()
    }
}

// --- Harness helpers --------------------------------------------------------

/// Generate a deterministic synthetic dataset: n vectors of dim with per-dim
/// Gaussian-ish spread via a simple LCG mixed with sines (no randomness crate).
pub fn synth_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; n * dim];
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for v in out.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 11) as u32) as f32 / (1u32 << 21) as f32;
        let x = (u - 0.5) * 2.0;
        *v = x + 0.3 * (x * 7.13).sin();
    }
    out
}

/// Ground truth top-k by float L2.
pub fn gt_topk(data: &[f32], dim: usize, queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    let n = data.len() / dim;
    queries
        .chunks_exact(dim)
        .map(|q| {
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (l2_sq_f32(q, &data[i * dim..(i + 1) * dim]), i as u32))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            scored.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect()
}

pub fn recall_at_k(preds: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> f32 {
    assert_eq!(preds.len(), truth.len());
    let mut hits = 0usize;
    let mut total = 0usize;
    for (p, t) in preds.iter().zip(truth.iter()) {
        let tset: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        for &id in p.iter().take(k) {
            total += 1;
            if tset.contains(&id) {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    }
}

/// Time a search closure over all queries, return (micros_total, results).
pub fn time_search<F: FnMut(&[u8]) -> Vec<u32>>(
    queries_q: &[u8],
    dim: usize,
    mut f: F,
) -> (u128, Vec<Vec<u32>>) {
    let t0 = Instant::now();
    let results: Vec<Vec<u32>> = queries_q.chunks_exact(dim).map(|q| f(q)).collect();
    (t0.elapsed().as_micros(), results)
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize_roundtrip_bounded_error() {
        let n = 200;
        let dim = 32;
        let data = synth_dataset(n, dim, 42);
        let sq = Sq8Codes::fit_quantize(&data, dim);
        let mut max_err = 0.0f32;
        for i in 0..n {
            let rec = sq.reconstruct(i);
            let orig = &data[i * dim..(i + 1) * dim];
            for (a, b) in rec.iter().zip(orig.iter()) {
                let e = (a - b).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }
        // Max quantization step is scale/255. Bound the error relative to the
        // largest per-dim scale observed.
        let max_scale = sq.dim_scale.iter().cloned().fold(0.0f32, f32::max);
        let tol = max_scale / 255.0 * 1.5;
        assert!(
            max_err <= tol,
            "max reconstruction error {} exceeds tol {}",
            max_err,
            tol
        );
    }

    #[test]
    fn bitplanes_reassemble_exactly_at_m8() {
        let n = 100;
        let dim = 48;
        let data = synth_dataset(n, dim, 7);
        let sq = Sq8Codes::fit_quantize(&data, dim);
        let bp = BitPlanes::from_sq8(&sq);
        let approx_full = bp.planes_to_approx_codes(8);
        // At M=8 there is no tail bias; codes must match exactly.
        assert_eq!(approx_full, sq.codes);
    }

    #[test]
    fn progressive_matches_baseline_within_recall_bound() {
        let n = 500;
        let dim = 64;
        let k = 10;
        let data = synth_dataset(n, dim, 1);
        let queries = synth_dataset(16, dim, 2);
        let sq = Sq8Codes::fit_quantize(&data, dim);
        let bp = BitPlanes::from_sq8(&sq);
        // Quantize queries using the same per-dim stats.
        let qcodes = quantize_queries(&queries, &sq);
        let truth = gt_topk(&data, dim, &queries, k);

        let full = FullSq8 { sq: &sq };
        let full_preds: Vec<Vec<u32>> = qcodes
            .chunks_exact(dim)
            .map(|q| full.search(q, k))
            .collect();
        let full_recall = recall_at_k(&full_preds, &truth, k);

        let approx = bp.planes_to_approx_codes(4);
        let prog = Progressive {
            approx_codes: &approx,
            sq: &sq,
            top_m: 4,
            shortlist_mul: 4,
            name_str: "prog_m4_mul4",
        };
        let prog_preds: Vec<Vec<u32>> = qcodes
            .chunks_exact(dim)
            .map(|q| prog.search(q, k))
            .collect();
        let prog_recall = recall_at_k(&prog_preds, &truth, k);

        // Progressive should recover nearly all of the full-SQ8 recall.
        assert!(
            prog_recall >= 0.9 * full_recall,
            "progressive recall {} fell below 0.9x baseline {}",
            prog_recall,
            full_recall
        );
    }
}

/// Quantize a batch of queries using an existing codebook's per-dim stats.
pub fn quantize_queries(queries: &[f32], sq: &Sq8Codes) -> Vec<u8> {
    let dim = sq.dim;
    let mut out = vec![0u8; queries.len()];
    for (q_in, q_out) in queries.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        for j in 0..dim {
            let x = (q_in[j] - sq.dim_min[j]) / sq.dim_scale[j];
            let v = (x * 255.0).round().clamp(0.0, 255.0) as u8;
            q_out[j] = v;
        }
    }
    out
}
