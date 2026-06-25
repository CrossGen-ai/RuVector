//! Shared NSW-style graph builder used by both [`crate::post_filter`] and
//! [`crate::serf`]. Construction is a brute-force k-NN graph followed by
//! reciprocal-edge merging, kept small and honest. The point of this crate is
//! to compare *search-time* strategies on the same graph topology.

use rayon::prelude::*;

use crate::data::{sq_l2, Point};

/// Adjacency list: `neighbors[i]` is the sorted neighborhood of node `i`.
#[derive(Debug, Clone)]
pub struct Graph {
    pub neighbors: Vec<Vec<u32>>,
}

impl Graph {
    /// Build a symmetric k-NN graph over `points` (brute force; O(n^2 d) which
    /// is fine for the n ≤ 10k sizes used in the benchmark). All edges are
    /// distinct and stored in both directions.
    pub fn build_knn(points: &[Point], k: usize) -> Self {
        let n = points.len();
        let raw: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let pi = &points[i].vec;
                let mut cands: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (sq_l2(pi, &points[j].vec), j as u32))
                    .collect();
                cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                cands.into_iter().take(k).map(|(_, j)| j).collect()
            })
            .collect();

        // Symmetrize: ensure (u,v) implies (v,u).
        let mut neighbors: Vec<Vec<u32>> = raw;
        let mut additions: Vec<Vec<u32>> = vec![Vec::new(); n];
        for u in 0..n {
            for &v in &neighbors[u] {
                additions[v as usize].push(u as u32);
            }
        }
        for u in 0..n {
            for v in additions[u].drain(..) {
                if !neighbors[u].contains(&v) {
                    neighbors[u].push(v);
                }
            }
            neighbors[u].sort_unstable();
            neighbors[u].dedup();
        }
        Self { neighbors }
    }

    /// Resident bytes for the adjacency lists. Each `u32` is 4 bytes; we add
    /// `Vec` header overhead (24 bytes on 64-bit). Uses `len()` not
    /// `capacity()` so the number reflects the data, not over-allocation
    /// pressure from intermediate growth during construction.
    pub fn estimated_bytes(&self) -> usize {
        let header = self.neighbors.len() * std::mem::size_of::<Vec<u32>>();
        let edges: usize = self.neighbors.iter().map(|v| v.len() * 4).sum();
        header + edges
    }

    pub fn num_edges(&self) -> usize {
        self.neighbors.iter().map(|v| v.len()).sum::<usize>() / 2
    }
}

/// Mini binary-heap-style ordered list of (distance, id) capped at `k`.
/// Returns hits sorted ascending by distance.
pub fn top_k(mut cands: Vec<(f32, u32)>, k: usize) -> Vec<(f32, u32)> {
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    cands.truncate(k);
    cands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;

    #[test]
    fn knn_graph_is_symmetric() {
        let ds = Dataset::random_gaussian(200, 8, 1);
        let g = Graph::build_knn(&ds.points, 8);
        for u in 0..ds.len() {
            for &v in &g.neighbors[u] {
                assert!(
                    g.neighbors[v as usize].contains(&(u as u32)),
                    "edge ({u},{v}) not reciprocated"
                );
            }
        }
    }

    #[test]
    fn knn_graph_neighbors_are_close() {
        let ds = Dataset::random_gaussian(200, 8, 2);
        let g = Graph::build_knn(&ds.points, 4);
        // Self-consistency: own neighbors should be closer than a uniform
        // random sample, on average.
        let u = 7;
        let mean_nbr = g.neighbors[u]
            .iter()
            .map(|&v| sq_l2(&ds.points[u].vec, &ds.points[v as usize].vec))
            .sum::<f32>()
            / g.neighbors[u].len() as f32;
        let mean_random = (0..ds.len())
            .step_by(11)
            .map(|j| sq_l2(&ds.points[u].vec, &ds.points[j].vec))
            .sum::<f32>()
            / ((ds.len() + 10) / 11) as f32;
        assert!(
            mean_nbr < mean_random,
            "neighbors should be closer than random ({mean_nbr} vs {mean_random})"
        );
    }
}
