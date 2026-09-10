//! # ruvector-lvq
//!
//! Locally-adaptive Vector Quantization (LVQ). Each vector is quantized with
//! its **own** per-vector scale and bias, computed from the vector's own
//! (dimension-wise) min/max. This preserves recall dramatically better than
//! a global scalar quantizer at the same bitwidth, because per-vector range
//! is typically far tighter than the global range across a corpus.
//!
//! Reference: Aguerrebere, Bhati, Hildebrand, Tepper, Willke — "Similarity
//! Search in the Blink of an Eye with Compressed Indices", VLDB 2023
//! (Intel Scalable Vector Search / SVS-LVQ). This crate is an independent
//! Rust reimplementation aimed at clarity, testability, and asymmetric
//! (query-fp32 vs code-int) distance computation.
//!
//! ## Variants
//! - `Lvq8` — 8 bits per component (4x compression from fp32).
//! - `Lvq4` — 4 bits per component (8x compression), packed two-per-byte.
//! - `Lvq4x8` — 4-bit primary code + 8-bit residual, giving a two-stage
//!   asymmetric distance: cheap 4-bit for candidate generation, refine
//!   with the residual for the top-k. Total ~1.5x of Lvq8 memory, recall
//!   near-fp32.
//!
//! ## Design
//! Every `Quantizer` produces a `Code` (owned bytes + per-vector metadata),
//! provides `encode(&[f32]) -> Code`, `decode(&Code) -> Vec<f32>`, and
//! `asymmetric_l2_sq(query: &[f32], code: &Code) -> f32`.
//!
//! The asymmetric distance dequantizes the code on the fly per component
//! and accumulates squared error against the fp32 query. This is the
//! standard SVS-LVQ inner loop. A SIMD implementation is left for a
//! follow-up; the scalar version here is what the benchmarks measure.

pub mod lvq4;
pub mod lvq8;
pub mod residual;

pub use lvq4::{Lvq4, Lvq4Code};
pub use lvq8::{Lvq8, Lvq8Code};
pub use residual::{Lvq4x8, Lvq4x8Code};

/// Common trait for locally-adaptive scalar quantizers.
///
/// Implementors define:
/// - `Code` — the wire type produced by `encode`.
/// - `encode(v)` — per-vector locally-adaptive quantization.
/// - `decode(c)` — approximate reconstruction (for tests / diagnostics).
/// - `asymmetric_l2_sq(q, c)` — squared L2 between an fp32 query and a
///   quantized code, without materializing the reconstruction.
pub trait Quantizer {
    type Code;
    fn dim(&self) -> usize;
    fn encode(&self, v: &[f32]) -> Self::Code;
    fn decode(&self, c: &Self::Code) -> Vec<f32>;
    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32;

    /// Estimated on-disk / on-heap bytes per code (excluding fixed overhead).
    fn bytes_per_code(&self) -> usize;
}

/// Exact fp32 L2 squared. Used as ground truth in tests and benchmarks.
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Estimate the top-1 recall of an approximate distance function against
/// exact fp32 L2 distance, on a uniformly random query set.
///
/// `approx_dist(q_idx, base_idx) -> f32`. Both functions must be defined
/// over the same query / base indices. Recall@k is measured as: fraction
/// of queries for which the approx top-1 base index is contained in the
/// exact top-k.
pub fn recall_at_k<F>(
    n_queries: usize,
    n_base: usize,
    exact: &dyn Fn(usize, usize) -> f32,
    approx: F,
    k: usize,
) -> f32
where
    F: Fn(usize, usize) -> f32,
{
    let mut hit = 0usize;
    for q in 0..n_queries {
        let mut exact_dists: Vec<(usize, f32)> =
            (0..n_base).map(|b| (b, exact(q, b))).collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let topk: std::collections::HashSet<usize> =
            exact_dists.iter().take(k).map(|(i, _)| *i).collect();

        let mut approx_dists: Vec<(usize, f32)> =
            (0..n_base).map(|b| (b, approx(q, b))).collect();
        approx_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let approx_top1 = approx_dists[0].0;

        if topk.contains(&approx_top1) {
            hit += 1;
        }
    }
    hit as f32 / n_queries as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_matches_hand_computation() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 0.0, 6.0];
        // (0)^2 + (2)^2 + (-3)^2 = 0 + 4 + 9 = 13
        assert!((l2_sq_f32(&a, &b) - 13.0).abs() < 1e-6);
    }
}
