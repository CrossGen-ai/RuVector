#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! # ruvector-nsg — Navigating Spreading-out Graph for ANN search
//!
//! Rust implementation of NSG (Fu, Xiang, Wang & Cai, *"Fast Approximate
//! Nearest Neighbor Search With The Navigating Spreading-out Graph"*,
//! VLDB 2019, <https://arxiv.org/abs/1707.00143>).
//!
//! ## Pipeline
//!
//! 1. **NN-Descent kNN graph init** (Dong, Charikar & Li, WWW 2011). Each
//!    point starts with K random neighbors. Iteratively explore "neighbors of
//!    neighbors" (local-join) and keep the K closest. Converges in O(log N)
//!    sweeps under the local-similarity assumption.
//! 2. **Navigating node selection.** Compute dataset centroid, then run an
//!    inexpensive greedy walk on the kNN graph from a random seed to the
//!    nearest base point to the centroid. That base point is the *fixed*
//!    entry point used by every query.
//! 3. **MRNG edge selection.** For each node `p`, expand the kNN graph by
//!    running a greedy beam search from the navigating node toward `p` to
//!    collect a candidate pool `C`. Sort `C` by distance to `p`, then apply
//!    the Monotonic Relative Neighborhood Graph pruning rule: keep edge
//!    `p → r` iff *no already-selected* neighbor `s` satisfies
//!    `d(s, r) < d(p, r)`. Cap out-degree at `R`.
//! 4. **DFS tree augmentation.** Run DFS from the navigating node. Any
//!    unreached vertex `v` gets an edge from its closest reached ancestor
//!    so the final graph is monotonically connected.
//!
//! Query is a single-layer greedy beam search from the fixed navigating
//! node with search list size `L_search` ≥ `k`.
//!
//! ## Why NSG vs HNSW
//!
//! NSG keeps **one** graph (no layers), uses MRNG pruning to guarantee a
//! monotonic search path, and typically reports lower memory + comparable
//! or better recall/QPS on million-scale static workloads. The tradeoff:
//! batch-only construction (no online insert). See ADR-195 for measured
//! numbers on this implementation.
//!
//! ## Determinism & safety
//!
//! - No `unsafe`, no BLAS/LAPACK, no C deps.
//! - Single-threaded build path is fully deterministic from `(seed, data)`.
//!   The optional rayon-parallel kNN construction is *not* bit-identical
//!   across thread counts (intentional, for speed); use
//!   [`NsgBuilder::deterministic(true)`] to force sequential build.

pub mod error;
pub mod knn;
pub mod nsg;

pub use error::{NsgError, Result};
pub use nsg::{NsgBuilder, NsgIndex, NsgParams, SearchHit};

/// Squared L2 distance — sufficient for ranking, avoids `sqrt`.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Brute-force top-k by squared L2. Used as the ground-truth reference for
/// recall measurement and the simplest baseline in benchmarks.
pub fn brute_force_topk(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<SearchHit> {
    let mut all: Vec<SearchHit> = (0..data.len())
        .map(|i| SearchHit {
            id: i as u32,
            dist: l2_sq(&data[i], query),
        })
        .collect();
    let k = k.min(all.len());
    // Partial sort is enough.
    all.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| {
        a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(k);
    all.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    all
}

/// Recall@k = |groundtruth ∩ predicted| / k.
pub fn recall_at_k(groundtruth: &[SearchHit], predicted: &[SearchHit]) -> f32 {
    if groundtruth.is_empty() {
        return 0.0;
    }
    let gt: std::collections::HashSet<u32> = groundtruth.iter().map(|h| h.id).collect();
    let hits = predicted.iter().filter(|h| gt.contains(&h.id)).count();
    hits as f32 / groundtruth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_basic() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0];
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn brute_force_returns_sorted() {
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![1.0, 1.0],
            vec![3.0, 3.0],
        ];
        let q = [0.0, 0.0];
        let r = brute_force_topk(&data, &q, 3);
        assert_eq!(r[0].id, 0);
        assert_eq!(r[1].id, 2);
        assert_eq!(r[2].id, 3);
        // Monotone in distance
        assert!(r[0].dist <= r[1].dist);
        assert!(r[1].dist <= r[2].dist);
    }

    #[test]
    fn recall_metric_exact() {
        let gt = vec![
            SearchHit { id: 1, dist: 0.1 },
            SearchHit { id: 2, dist: 0.2 },
            SearchHit { id: 3, dist: 0.3 },
        ];
        let pred = vec![
            SearchHit { id: 1, dist: 0.1 },
            SearchHit { id: 9, dist: 0.15 },
            SearchHit { id: 3, dist: 0.3 },
        ];
        assert!((recall_at_k(&gt, &pred) - (2.0 / 3.0)).abs() < 1e-6);
    }
}
