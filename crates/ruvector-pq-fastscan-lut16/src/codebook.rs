//! Shared PQ codebook training and encoding.

use crate::{PqError, KS16};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A per-sub-quantizer codebook. `centroids[c][j]` = the j-th coordinate of
/// the c-th centroid in the sub-space owned by this sub-quantizer.
#[derive(Debug, Clone)]
pub struct SubCodebook {
    pub centroids: Vec<Vec<f32>>, // [ks][ds]
    pub ds: usize,
}

impl SubCodebook {
    pub fn ks(&self) -> usize {
        self.centroids.len()
    }
}

/// Train `m` sub-codebooks by running k-means separately on each sub-space.
pub fn train_codebook(
    data: &[f32],
    n: usize,
    d: usize,
    m: usize,
    ks: usize,
    iters: usize,
    seed: u64,
) -> Result<Vec<SubCodebook>, PqError> {
    if d % m != 0 {
        return Err(PqError::BadShape { d, m });
    }
    if n == 0 {
        return Err(PqError::EmptyTraining);
    }
    let ds = d / m;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut books = Vec::with_capacity(m);
    for sq in 0..m {
        let mut sub: Vec<f32> = Vec::with_capacity(n * ds);
        for i in 0..n {
            let off = i * d + sq * ds;
            sub.extend_from_slice(&data[off..off + ds]);
        }
        let centroids = kmeans_lloyd(&sub, n, ds, ks, iters, &mut rng);
        books.push(SubCodebook { centroids, ds });
        let _ = sq;
    }
    Ok(books)
}

fn kmeans_lloyd(
    data: &[f32],
    n: usize,
    d: usize,
    ks: usize,
    iters: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = (0..ks)
        .map(|_| {
            let pick = rng.gen_range(0..n);
            data[pick * d..(pick + 1) * d].to_vec()
        })
        .collect();
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let dist = sq_l2(x, cent);
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }
        let mut sums = vec![vec![0f32; d]; ks];
        let mut counts = vec![0u32; ks];
        for i in 0..n {
            let c = assign[i] as usize;
            let x = &data[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c][j] += x[j];
            }
            counts[c] += 1;
        }
        for c in 0..ks {
            if counts[c] == 0 {
                let pick = rng.gen_range(0..n);
                centroids[c] = data[pick * d..(pick + 1) * d].to_vec();
            } else {
                let cnt = counts[c] as f32;
                for j in 0..d {
                    centroids[c][j] = sums[c][j] / cnt;
                }
            }
        }
    }
    centroids
}

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Encode one vector -> per-sub-quantizer nearest-centroid ids.
pub fn encode_one(x: &[f32], books: &[SubCodebook]) -> Vec<u8> {
    let m = books.len();
    let mut codes = vec![0u8; m];
    let mut off = 0usize;
    for sq in 0..m {
        let book = &books[sq];
        let ds = book.ds;
        let sub = &x[off..off + ds];
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for (c, cent) in book.centroids.iter().enumerate() {
            let dist = sq_l2(sub, cent);
            if dist < best_d {
                best_d = dist;
                best = c as u8;
            }
        }
        codes[sq] = best;
        off += ds;
    }
    codes
}

/// Per-sub-quantizer u8-quantized LUT for LUT16 FastScan.
pub struct Lut16U8 {
    pub lut: Vec<u8>,
    pub scale: f32,
}

/// Build the u8 LUT for LUT16 FastScan.
///
/// The scaling scheme:
///   1. Compute f32 tables `t[sq][c] = ||q_sub - centroid||^2`.
///   2. `t_max = max over all (sq, c)`.
///   3. Quantize `u8_val = round(t[sq][c] / t_max * 255/m)` so a sum over
///      `m` sub-quantizers fits in u8 * m <= 255, keeping u16 accumulators
///      safe.
///   4. `scale = t_max * m / 255` recovers f32 distance.
pub fn build_lut16_u8(books: &[SubCodebook], query: &[f32]) -> Lut16U8 {
    let m = books.len();
    let mut t = vec![0f32; m * KS16];
    let mut off = 0usize;
    for sq in 0..m {
        let ds = books[sq].ds;
        let qsub = &query[off..off + ds];
        for c in 0..KS16 {
            t[sq * KS16 + c] = sq_l2(qsub, &books[sq].centroids[c]);
        }
        off += ds;
    }
    let t_max = t.iter().cloned().fold(0f32, f32::max).max(1e-30);
    let per_sq_max = 255.0f32 / m as f32;
    let mut lut = vec![0u8; m * KS16];
    for i in 0..lut.len() {
        let q = (t[i] / t_max * per_sq_max).round();
        lut[i] = q.clamp(0.0, per_sq_max) as u8;
    }
    Lut16U8 {
        lut,
        scale: t_max * m as f32 / 255.0,
    }
}

/// Top-k by ascending distance.
pub fn top_k(mut v: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let k = k.min(v.len());
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(k);
    v
}

/// Exact brute-force top-k over row-major `[n * d]` corpus.
pub fn exact_topk(data: &[f32], n: usize, d: usize, query: &[f32], k: usize) -> Vec<u32> {
    let mut v: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, sq_l2(query, &data[i * d..(i + 1) * d])))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(k);
    v.into_iter().map(|(id, _)| id).collect()
}

/// Recall of candidate set against truth set (fraction of truth that appears
/// in candidate).
pub fn recall_at_k(truth: &[u32], candidate: &[u32]) -> f32 {
    let k = truth.len().max(1);
    let mut hit = 0;
    for c in candidate {
        if truth.contains(c) {
            hit += 1;
        }
    }
    hit as f32 / k as f32
}
