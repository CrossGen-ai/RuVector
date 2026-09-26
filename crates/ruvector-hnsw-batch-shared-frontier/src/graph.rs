//! Minimal proximity graph used as an HNSW stand-in.
//!
//! We build a single-layer approximate k-NN graph: for each node, keep the
//! `m` nearest neighbours (by squared-L2) among a random candidate pool
//! of size `pool`. This is deliberately small — not competitive with a
//! real HNSW build — but produces a navigable graph whose greedy-search
//! access pattern matches HNSW's lower layer, which is what this
//! experiment measures.
//!
//! Vector store is a single contiguous `Vec<f32>` so `vector(id)` returns
//! a plain slice — friendly to the batched distance kernel.

use crate::dist::l2_sq;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A flat proximity graph over `n` vectors of dimension `dim`.
pub struct ProximityGraph {
    pub dim: usize,
    pub n: usize,
    pub m: usize,
    data: Vec<f32>,
    /// `neighbours[i]` is a fixed-length slice of length `m`.
    neighbours: Vec<u32>,
    /// Entry point used to seed greedy search.
    pub entry: u32,
}

impl ProximityGraph {
    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    #[inline]
    pub fn neighbours(&self, id: u32) -> &[u32] {
        let i = id as usize * self.m;
        &self.neighbours[i..i + self.m]
    }

    /// Exact brute-force top-`k`, used to compute ground-truth recall.
    pub fn brute_top_k(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut all: Vec<(f32, u32)> = (0..self.n as u32)
            .map(|id| (l2_sq(query, self.vector(id)), id))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

pub struct ProximityGraphBuilder {
    pub n: usize,
    pub dim: usize,
    pub m: usize,
    pub pool: usize,
    pub seed: u64,
}

impl ProximityGraphBuilder {
    pub fn build(&self, data: Vec<f32>) -> ProximityGraph {
        assert_eq!(data.len(), self.n * self.dim, "data / n*dim mismatch");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut neighbours = vec![u32::MAX; self.n * self.m];
        let ids: Vec<u32> = (0..self.n as u32).collect();

        for i in 0..self.n {
            // Sample a candidate pool for this node.
            let mut pool_ids: Vec<u32> = ids.clone();
            pool_ids.shuffle(&mut rng);
            pool_ids.retain(|&x| x != i as u32);
            pool_ids.truncate(self.pool);

            let vi_start = i * self.dim;
            let vi = &data[vi_start..vi_start + self.dim];
            let mut scored: Vec<(f32, u32)> = pool_ids
                .iter()
                .map(|&j| {
                    let js = j as usize * self.dim;
                    let vj = &data[js..js + self.dim];
                    (l2_sq(vi, vj), j)
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (slot, (_, id)) in scored.iter().take(self.m).enumerate() {
                neighbours[i * self.m + slot] = *id;
            }
            // Fill any unused slots with the last kept id (rare when pool >= m).
            let filled = scored.len().min(self.m);
            for slot in filled..self.m {
                neighbours[i * self.m + slot] = neighbours[i * self.m + filled - 1];
            }
        }

        ProximityGraph {
            dim: self.dim,
            n: self.n,
            m: self.m,
            data,
            neighbours,
            entry: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn builds_and_queries() {
        let n = 200; let dim = 16; let m = 8;
        let data = synth(n, dim, 42);
        let g = ProximityGraphBuilder { n, dim, m, pool: 64, seed: 1 }.build(data);
        assert_eq!(g.n, n);
        assert_eq!(g.vector(3).len(), dim);
        assert_eq!(g.neighbours(3).len(), m);
        let q: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let top = g.brute_top_k(&q, 5);
        assert_eq!(top.len(), 5);
    }
}
