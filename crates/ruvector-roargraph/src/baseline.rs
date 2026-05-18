//! Base-to-base k-NN graph baseline ("OOD-naive" comparison).
//!
//! Builds a graph where each base vector is connected to its k nearest
//! neighbours among *other base vectors* (using exact L2).  This is the
//! strategy employed by HNSW/NSG at layer 0 and represents the natural
//! approach when no query distribution information is available.
//!
//! When queries are OOD (drawn from a different distribution than the base
//! corpus), this graph is poorly oriented toward the actual query entry
//! regions — RoarGraph's bipartite projection exploits query information
//! to construct a better-aligned graph.

use crate::error::RoarError;
use crate::graph::{l2sq, RoarGraph, SearchResult};
use crate::AnnIndex;

/// A base-to-base k-NN graph index (the "OOD-naive" baseline).
pub struct BaselineGraph {
    inner: RoarGraph,
    max_degree: usize,
}

impl BaselineGraph {
    /// Create a new baseline graph.
    ///
    /// `max_degree` is the number of base-to-base neighbours retained per node.
    pub fn new(dim: usize, max_degree: usize) -> Self {
        BaselineGraph {
            inner: RoarGraph::new(dim),
            max_degree,
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl AnnIndex for BaselineGraph {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RoarError> {
        self.inner.add(vectors)
    }

    /// Build the base-to-base k-NN graph.
    ///
    /// For each node v, we compute its distance to every other base node and
    /// retain the `max_degree` closest.  O(n² · d) — fine for n=5 000.
    fn build(&mut self, _training_queries: &[Vec<f32>]) -> Result<(), RoarError> {
        if self.inner.vectors.is_empty() {
            return Err(RoarError::EmptyIndex);
        }
        let n = self.inner.vectors.len();
        let k = self.max_degree.min(n - 1);

        let mut adj: Vec<Vec<u32>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, l2sq(&self.inner.vectors[i], &self.inner.vectors[j])))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            adj.push(dists.into_iter().map(|(id, _)| id).collect());
        }

        self.inner.adj = adj;
        self.inner.built = true;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, RoarError> {
        self.inner.search(query, k, ef)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_baseline_build_and_search() {
        let base = random_vecs(50, 8, 42);
        let mut idx = BaselineGraph::new(8, 10);
        idx.add(&base).unwrap();
        idx.build(&[]).unwrap();
        let q = base[0].clone();
        let results = idx.search(&q, 5, 20).unwrap();
        assert!(!results.is_empty());
        // first result should be the query itself (id=0, dist≈0)
        assert_eq!(results[0].id, 0);
        assert!(results[0].dist < 1e-6);
    }

    #[test]
    fn test_baseline_each_node_has_neighbours() {
        let base = random_vecs(30, 4, 7);
        let mut idx = BaselineGraph::new(4, 5);
        idx.add(&base).unwrap();
        idx.build(&[]).unwrap();
        for nb_list in idx.inner.adj.iter() {
            assert!(!nb_list.is_empty(), "every node should have at least one neighbour");
        }
    }
}
