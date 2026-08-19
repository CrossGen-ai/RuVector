//! Small k-NN graph built by brute force. This mimics the *topology* of an
//! NSW / HNSW layer 0 (each node has ~M nearest neighbors) which is what the
//! beam-search termination policy actually sees; it does not aim to be a
//! competitive index. The point of the PoC is to compare termination
//! policies on identical traversal dynamics.

use crate::util::sq_l2;
use crate::Scored;

/// Row-major dense-vector store + directed k-NN adjacency list.
#[derive(Debug)]
pub struct KnnGraph {
    pub dim: usize,
    pub n: usize,
    /// `n * dim` f32s, row-major.
    pub data: Vec<f32>,
    /// `n * m` u32 neighbor ids (each row is one node's neighbor list,
    /// sorted by distance ascending — this is what an HNSW-like beam
    /// search would see after `select_neighbors_heuristic`).
    pub adj: Vec<u32>,
    pub m: usize,
}

impl KnnGraph {
    /// Build the graph by exact brute-force k-NN. O(n^2 dim). Fine for
    /// PoC-scale n (≤ 20k); the whole point is to hold traversal
    /// dynamics fixed while we vary the termination policy.
    pub fn build(data: Vec<f32>, dim: usize, m: usize) -> Self {
        assert!(dim > 0 && m > 0);
        assert_eq!(data.len() % dim, 0);
        let n = data.len() / dim;
        assert!(m < n, "graph degree m={m} must be < n={n}");

        let mut adj = vec![0u32; n * m];
        // Per-node scratch heap keeping the m smallest distances.
        for i in 0..n {
            let vi = &data[i * dim..(i + 1) * dim];
            let mut cand: Vec<Scored> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = sq_l2(vi, &data[j * dim..(j + 1) * dim]);
                cand.push(Scored { dist: d, id: j as u32 });
            }
            cand.sort();
            for k in 0..m {
                adj[i * m + k] = cand[k].id;
            }
        }
        Self { dim, n, data, adj, m }
    }

    #[inline]
    pub fn vec(&self, id: u32) -> &[f32] {
        let id = id as usize;
        &self.data[id * self.dim..(id + 1) * self.dim]
    }

    #[inline]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let id = id as usize;
        &self.adj[id * self.m..(id + 1) * self.m]
    }

    /// Exact top-k by brute force. Used to compute ground-truth recall in
    /// the benchmark — not used during timed search.
    pub fn brute_topk(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut all: Vec<Scored> = (0..self.n as u32)
            .map(|id| Scored { dist: sq_l2(query, self.vec(id)), id })
            .collect();
        all.sort();
        all.iter().take(k).map(|s| s.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::random_unit_vectors;

    #[test]
    fn graph_has_correct_shape() {
        let data = random_unit_vectors(50, 8, 1);
        let g = KnnGraph::build(data, 8, 6);
        assert_eq!(g.n, 50);
        assert_eq!(g.adj.len(), 50 * 6);
        for id in 0..50u32 {
            for &nb in g.neighbors(id) {
                assert_ne!(nb, id, "no self-loops");
            }
        }
    }

    #[test]
    fn neighbors_are_sorted_by_distance() {
        let data = random_unit_vectors(30, 4, 2);
        let g = KnnGraph::build(data, 4, 5);
        for id in 0..30u32 {
            let nbrs = g.neighbors(id);
            let dists: Vec<f32> = nbrs.iter().map(|&j| sq_l2(g.vec(id), g.vec(j))).collect();
            for w in dists.windows(2) {
                assert!(w[0] <= w[1] + 1e-6, "neighbors not sorted: {dists:?}");
            }
        }
    }

    #[test]
    fn brute_topk_returns_k_ids() {
        let data = random_unit_vectors(20, 8, 5);
        let g = KnnGraph::build(data.clone(), 8, 5);
        let q = &data[0..8];
        let ids = g.brute_topk(q, 5);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], 0, "closest to a stored vector is itself");
    }
}
