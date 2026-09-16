//! ruvector-aq-beam
//!
//! Additive Quantization (AQ) with beam-search encoding, plus a Product
//! Quantization (PQ) baseline for apples-to-apples comparison.
//!
//! AQ approximates a vector `x` in R^d as a sum of `M` centroids, one drawn
//! from each of `M` codebooks `C_m` of shape `K x d`:
//!
//! ```text
//! x ~= sum_{m=1..M} C_m[i_m],    i_m in [0, K).
//! ```
//!
//! Because each codebook spans the full dimensionality (unlike PQ which
//! carves R^d into disjoint sub-spaces), AQ can express correlations across
//! all dimensions and typically reaches lower reconstruction error at the
//! same bitrate `M · log2(K)`. Encoding is NP-hard in general; we use beam
//! search over codebooks in a fixed order (Babenko & Lempitsky, CVPR 2014).
//!
//! Distance queries under L2 use an ADC-style precomputed table: for a
//! query q, we compute `t_m[k] = ‖q‖² · [m==0] − 2 q·c_m[k] + ‖c_m[k]‖² +
//! 2 Σ_{m'<m} c_m[k]·c_{m'}[i_{m'}]` — but the naive form suffices for
//! this PoC: expand ‖q − x̂‖² and sum four table lookups per code.

pub mod kmeans;
pub mod pq;
pub mod aq;

pub use aq::{AqBeamIndex, AqParams};
pub use pq::{PqIndex, PqParams};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AqError {
    #[error("invalid dimension: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("K must be a power of 2 and > 0")]
    InvalidK,
    #[error("M must be > 0")]
    InvalidM,
}

/// Common interface across quantizers so the benchmark can iterate them.
pub trait Quantizer: Sync {
    fn dim(&self) -> usize;
    fn code_bytes(&self) -> usize;
    fn name(&self) -> &'static str;

    /// Encode a raw dataset into compact codes (one row per vector).
    fn encode_dataset(&self, xs: &[Vec<f32>]) -> Vec<Vec<u8>>;

    /// Reconstruct a vector from its code.
    fn reconstruct(&self, code: &[u8]) -> Vec<f32>;

    /// Exhaustive top-`k` search under squared-L2 distance. Returns
    /// (index, distance) pairs sorted ascending by distance.
    fn search(&self, query: &[f32], codes: &[Vec<u8>], k: usize) -> Vec<(usize, f32)>;
}

/// Squared L2 distance between two equal-length slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Ground-truth exhaustive search on raw f32 vectors — used for recall.
pub fn brute_force_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut out: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, sq_l2(query, v)))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    out
}

/// Recall@k of `retrieved` w.r.t. `truth`, both length ≤ k.
pub fn recall_at_k(retrieved: &[(usize, f32)], truth: &[(usize, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<usize> = truth.iter().map(|(i, _)| *i).collect();
    let hit = retrieved.iter().filter(|(i, _)| truth_ids.contains(i)).count();
    hit as f32 / truth.len() as f32
}
