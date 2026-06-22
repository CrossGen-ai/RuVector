//! kNN-graph data structure and the `KnnGraphBuilder` trait.
//!
//! A `KnnGraph` is just `Vec<Vec<KnnNeighbor>>` — adjacency list with
//! distances cached. Builders fill it. The trait is the swap-point that
//! lets the benchmark compare algorithms without per-variant glue code.

use crate::distance::DistanceCounter;

#[derive(Clone, Copy, Debug)]
pub struct KnnNeighbor {
    pub id: u32,
    pub dist: f32,
}

impl KnnNeighbor {
    pub const SENTINEL: KnnNeighbor = KnnNeighbor { id: u32::MAX, dist: f32::INFINITY };
}

/// Adjacency list — `neighbors[i]` are the K closest points to point i,
/// sorted closest-first.
#[derive(Clone, Debug)]
pub struct KnnGraph {
    pub k: usize,
    pub neighbors: Vec<Vec<KnnNeighbor>>,
}

impl KnnGraph {
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            k,
            neighbors: vec![Vec::with_capacity(k); n],
        }
    }

    pub fn n(&self) -> usize {
        self.neighbors.len()
    }

    /// Recall@k versus a ground-truth graph: fraction of true top-k
    /// neighbors recovered, averaged over all points.
    ///
    /// Excludes self-edges in both sides.
    pub fn recall_against(&self, truth: &KnnGraph, k: usize) -> f64 {
        assert_eq!(self.n(), truth.n());
        let mut total = 0u64;
        let mut hits = 0u64;
        for i in 0..self.n() {
            let truth_set: std::collections::HashSet<u32> = truth.neighbors[i]
                .iter()
                .filter(|n| n.id as usize != i)
                .take(k)
                .map(|n| n.id)
                .collect();
            if truth_set.is_empty() {
                continue;
            }
            total += truth_set.len() as u64;
            for n in self.neighbors[i].iter().filter(|n| n.id as usize != i).take(k) {
                if truth_set.contains(&n.id) {
                    hits += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

/// Implemented by every bulk-construction backend (brute force, NN-Descent, …).
pub trait KnnGraphBuilder {
    fn name(&self) -> &'static str;
    fn build(&self, vectors: &[Vec<f32>], k: usize, counter: &DistanceCounter) -> KnnGraph;
}
