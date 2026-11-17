//! Subspace-Conditional Product Quantization (SC-PQ).
//!
//! Classic PQ splits a D-dim vector into M subvectors and quantizes each
//! with an independent codebook of K centroids. This loses information
//! whenever subspaces are statistically correlated, because the codebook
//! for subspace m cannot see which code was chosen for subspace m-1.
//!
//! SC-PQ addresses this by storing, for each subspace m>0 and each code
//! p in {0..K}, a dedicated sub-codebook C_{m,p} of K centroids trained
//! on residuals whose previous-subspace code is p. Encoding is greedy
//! left-to-right. At query time we precompute a per-subspace table
//! `lut_m[prev][cur] = dist(q_m, C_{m,prev}[cur])` of size K*K; the
//! per-vector scan then does M lookups keyed by (prev_code, cur_code).
//! Subspace m=0 uses a plain K-centroid codebook, so `lut_0[cur]` is the
//! standard PQ LUT.
//!
//! Scan cost per vector: identical to PQ (M table lookups + M adds).
//! LUT build cost: O(M*K*K*(D/M)) = O(D*K*K), modest for K=16 or 32.
//! Codebook memory: M*K*K*(D/M) = D*K*K floats (K× more than PQ), fine
//! for K<=32.
//!
//! The trait-based `Quantizer` design lets callers swap SC-PQ for the
//! baseline PQ (shipped here for apples-to-apples comparison) through a
//! common interface, so a production index can A/B the two backends.

#![forbid(unsafe_code)]

pub mod codebook;
pub mod data;
pub mod pq;
pub mod scpq;

pub use codebook::{train_kmeans, Codebook};
pub use pq::Pq;
pub use scpq::ScPq;

/// Common interface shared by PQ and SC-PQ (and any future variant).
pub trait Quantizer {
    /// Encode a dataset of N D-dim vectors into N*M byte codes (K<=256).
    fn encode(&self, data: &[f32]) -> Vec<u8>;
    /// Build per-query lookup tables so `score` can run.
    fn build_lut(&self, query: &[f32]) -> Vec<f32>;
    /// Scan all codes against a prebuilt LUT, returning squared-L2 scores.
    fn score(&self, lut: &[f32], codes: &[u8]) -> Vec<f32>;
    /// Number of subspaces M.
    fn m(&self) -> usize;
    /// Centroids per subspace K.
    fn k(&self) -> usize;
    /// Input dimensionality D.
    fn d(&self) -> usize;
}

/// Brute-force top-k over the scored vectors; used in benches.
pub fn top_k(scores: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as u32, s))
        .collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Exact squared-L2 brute-force top-k, used as the recall ground truth.
pub fn exact_top_k(query: &[f32], data: &[f32], d: usize, k: usize) -> Vec<(u32, f32)> {
    let n = data.len() / d;
    let mut scored = Vec::with_capacity(n);
    for i in 0..n {
        let v = &data[i * d..(i + 1) * d];
        let mut s = 0f32;
        for j in 0..d {
            let dx = query[j] - v[j];
            s += dx * dx;
        }
        scored.push((i as u32, s));
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Recall@k: fraction of the exact top-k retrieved in the approximate top-k.
pub fn recall_at_k(approx: &[(u32, f32)], exact: &[(u32, f32)]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let want: std::collections::HashSet<u32> = exact.iter().map(|x| x.0).collect();
    let hit = approx.iter().filter(|x| want.contains(&x.0)).count();
    hit as f32 / exact.len() as f32
}
