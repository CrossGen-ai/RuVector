//! ruvector-symphony-qg
//!
//! Co-designed graph + 1-bit quantization index. Inspired by SymphonyQG
//! (SIGMOD 2025, Gou et al.) — graph neighbours store inline binary codes
//! so candidate distance estimation during graph traversal happens at
//! popcount speed, with f32 reranking applied only to the top-K survivors.
//!
//! See `docs/research/nightly/2026-06-17-symphony-qg/README.md` for the
//! design rationale, benchmark methodology, and known divergences from
//! the paper.
//!
//! # Quick start
//!
//! ```ignore
//! use ruvector_symphony_qg::{SymphonyIndex, IndexParams};
//! let vectors: Vec<Vec<f32>> = load_data();
//! let idx = SymphonyIndex::build(&vectors, IndexParams::default());
//! let hits = idx.search(&query, 10);
//! ```

pub mod quant;
pub mod graph;
pub mod symphony;

pub use quant::BitQuantizer;
pub use graph::{KnnGraph, GraphParams};
pub use symphony::{SymphonyIndex, IndexParams, SearchStats};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("empty corpus")]
    Empty,
}

/// Squared L2 distance between two vectors.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Brute-force kNN (ground truth oracle for recall measurement).
pub fn brute_force_knn(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq(v, query)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

/// Recall@k of `got` against `truth` — fraction of truth ids found in got.
pub fn recall_at_k(got: &[(usize, f32)], truth: &[(usize, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<usize> = truth.iter().map(|x| x.0).collect();
    let hits = got.iter().filter(|x| truth_ids.contains(&x.0)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0, 4.0];
        assert!((l2_sq(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_basic() {
        let truth = vec![(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)];
        let got = vec![(1, 0.0), (5, 0.0), (3, 0.0), (4, 0.0)];
        assert!((recall_at_k(&got, &truth) - 0.75).abs() < 1e-6);
    }
}
