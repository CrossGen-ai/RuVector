//! Permutation-Prefix Reranker (PPR).
//!
//! Each vector is summarized by the ordered indices of its `m` nearest
//! pivots (a *permutation prefix*). Rerank cost per candidate is a small
//! integer-only Spearman-footrule distance between the candidate's prefix
//! and the query's prefix. Rerank storage per candidate is `m * ceil(log2(n_pivots))`
//! bits — typically 16-64 bytes.
//!
//! The crate exposes a swappable `Reranker` trait with three implementations:
//! full-L2 (ground truth), product-quantization asymmetric distance (PQ-ADC)
//! as a strong baseline, and permutation-prefix.

pub mod l2;
pub mod pivot;
pub mod ppr;
pub mod pq;

use std::time::Duration;

/// A reranker takes a query vector and a shortlist of candidate ids, returns the
/// candidates sorted ascending by its own distance proxy.
pub trait Reranker {
    fn name(&self) -> &'static str;
    fn bytes_per_vector(&self) -> usize;
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32>;
}

#[derive(Clone, Debug)]
pub struct VariantStats {
    pub name: String,
    pub bytes_per_vector: usize,
    pub build_time: Duration,
    pub mean_rerank_latency_us: f64,
    pub p50_rerank_latency_us: f64,
    pub p95_rerank_latency_us: f64,
    pub recall_at_k: f64,
}

/// L2 brute force distance (shared helper, not just for `l2` module).
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Evaluate recall@k given the retrieved ids (from a reranker) and the exact
/// top-k ground truth ids. Order within each set is irrelevant for recall.
pub fn recall_at_k(retrieved: &[u32], truth: &[u32]) -> f64 {
    let k = truth.len();
    if k == 0 {
        return 1.0;
    }
    let mut hits = 0usize;
    for id in retrieved.iter().take(k) {
        if truth.contains(id) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_matches_manual() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];
        // (1-4)^2 + (2-6)^2 + 0 = 9 + 16 = 25
        assert!((sq_l2(&a, &b) - 25.0).abs() < 1e-5);
    }

    #[test]
    fn recall_counts_set_overlap() {
        assert!((recall_at_k(&[1, 2, 3], &[1, 2, 4]) - 2.0 / 3.0).abs() < 1e-9);
        assert!((recall_at_k(&[], &[]) - 1.0).abs() < 1e-9);
    }
}
