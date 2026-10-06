//! Exact-kNN proximity graph used as the search substrate.
//!
//! We build the graph by brute-force k-nearest-neighbors over the dataset
//! (O(n^2 d), a few seconds for n up to ~50K in release). The graph is
//! therefore provably correct: every edge truly connects to one of the
//! node's m closest neighbors. In a production HNSW this role is played
//! by greedy insertion with a selection heuristic; the entry-selection
//! improvement being measured in this crate is orthogonal to the build
//! strategy, so holding the graph exact lets us isolate the entry effect
//! cleanly from noise introduced by approximate construction.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::metrics::l2_sq;

/// Immutable graph after build.
pub struct Nsw {
    pub dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub vectors: Vec<f32>, // row-major, len = n * dim
    pub neighbors: Vec<u32>, // row-major, len = n * m
    pub fixed_entry: u32,
    pub n: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct NswParams {
    pub m: usize,
    pub ef_construction: usize,
    /// Overwrite the last `long_edges` slots per node with random long
    /// edges. Simulates the small-world property real NSW/HNSW gets from
    /// greedy insertion; zero = pure kNN graph.
    pub long_edges: usize,
    pub long_edge_seed: u64,
}

impl Default for NswParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 64, long_edges: 0, long_edge_seed: 0xA5A5 }
    }
}

impl Nsw {
    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.vectors[i..i + self.dim]
    }

    #[inline]
    pub fn neighbors_of(&self, id: u32) -> &[u32] {
        let i = id as usize * self.m;
        &self.neighbors[i..i + self.m]
    }

    pub fn build(vectors: Vec<f32>, dim: usize, params: NswParams) -> Self {
        assert!(vectors.len() % dim == 0);
        let n = vectors.len() / dim;
        assert!(n >= 2, "graph needs at least 2 vectors");
        let m = params.m;
        let ef_c = params.ef_construction;
        let mut neighbors = vec![u32::MAX; n * m];

        for i in 0..n {
            let vi = &vectors[i * dim..(i + 1) * dim];
            let mut scored: Vec<(f32, u32)> = (0..n as u32)
                .filter(|&j| j as usize != i)
                .map(|j| {
                    (l2_sq(&vectors[j as usize * dim..(j as usize + 1) * dim], vi), j)
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (slot, (_, id)) in scored.iter().take(m).enumerate() {
                neighbors[i * m + slot] = *id;
            }
        }

        // Optionally overwrite the LAST `long_edges` neighbor slots per node
        // with uniformly random edges. This is a simple proxy for the long
        // links real NSW/HNSW builds carry from early insertions; it makes
        // the fixed-entry baseline competitive and surfaces the regime
        // where LSH-seeding is still beneficial.
        if params.long_edges > 0 && params.long_edges < m {
            let mut rng = StdRng::seed_from_u64(params.long_edge_seed);
            for i in 0..n {
                for slot in (m - params.long_edges)..m {
                    let mut j = rng.gen_range(0..n as u32);
                    if j as usize == i { j = ((i + 1) % n) as u32; }
                    neighbors[i * m + slot] = j;
                }
            }
        }

        Self { dim, m, ef_construction: ef_c, vectors, neighbors, fixed_entry: 0, n }
    }
}
