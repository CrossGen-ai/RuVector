//! NN-Descent: approximate k-NN graph construction via local-join refinement.
//!
//! Implements Dong, Charikar & Li, "Efficient k-Nearest Neighbor Graph
//! Construction for Generic Similarity Measures", WWW 2011, with the
//! sampling-rate, early-termination, and reverse-neighbor-join refinements
//! that survived into PyNNDescent and NVIDIA CAGRA's CPU build path.
//!
//! Design rules:
//! - Swappable [`Metric`] trait so backends (Euclidean, dot, custom) coexist.
//! - All algorithms produce the same output type: a flat `Vec<NeighborList>`
//!   so callers can drop the result into HNSW seeding, DiskANN candidate
//!   pools, or graph diagnostics without conversion.
//! - No mocks. The "exact" baseline really is exact and the bench compares
//!   against it for recall.

pub mod heap;
pub mod brute;
pub mod nndescent;

use std::time::Duration;

/// A k-NN graph: `graph[i]` is the (up to) `k` approximate neighbours of `i`,
/// sorted by ascending distance, *excluding* `i` itself.
pub type KnnGraph = Vec<Vec<Neighbor>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

impl Eq for Neighbor {}

/// Distance backend. Implementors must return a non-negative scalar where
/// "smaller is closer" — NN-Descent's local join is metric-agnostic but the
/// pruning step assumes ordering by ascending distance.
pub trait Metric: Sync {
    fn dist(&self, a: &[f32], b: &[f32]) -> f32;
}

pub struct L2;
impl Metric for L2 {
    #[inline]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s // squared L2 — ordering-equivalent to L2, faster
    }
}

/// Builder-side trait shared by brute-force and NN-Descent so callers can
/// swap algorithms without rewriting plumbing.
pub trait KnnGraphBuilder {
    fn build(&mut self, data: &[Vec<f32>], k: usize) -> BuildReport;
}

#[derive(Debug, Clone)]
pub struct BuildReport {
    pub graph: KnnGraph,
    pub elapsed: Duration,
    pub distance_calls: u64,
    pub iterations: u32,
}

/// Compute recall@k of `approx` against `truth`.
/// Both must have identical shape; `truth` is treated as ground-truth.
pub fn recall_at_k(truth: &KnnGraph, approx: &KnnGraph) -> f64 {
    assert_eq!(truth.len(), approx.len());
    let mut hits: u64 = 0;
    let mut total: u64 = 0;
    for (t, a) in truth.iter().zip(approx.iter()) {
        let t_ids: std::collections::HashSet<u32> = t.iter().map(|n| n.id).collect();
        for n in a {
            if t_ids.contains(&n.id) {
                hits += 1;
            }
        }
        total += t.len() as u64;
    }
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}
