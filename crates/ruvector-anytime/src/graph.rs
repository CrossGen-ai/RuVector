//! Flat navigable small-world proximity graph — HNSW layer-0 equivalent.
//!
//! Local exact k-NN edges plus random long-jump shortcuts (the "navigable"
//! property). Built brute-force in parallel; sufficient for the PoC.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;

/// Configuration for the proximity graph.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Local neighbors per node (exact k-NN).
    pub m: usize,
    /// Long-jump (random) neighbors per node — small-world shortcuts.
    pub m_longjump: usize,
    /// Dimensions per vector.
    pub dims: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            m: 16,
            m_longjump: 4,
            dims: 64,
        }
    }
}

/// Flat navigable small-world graph (single layer).
pub struct FlatGraph {
    vectors: Vec<f32>,
    pub neighbors: Vec<Vec<u32>>,
    pub config: GraphConfig,
    pub n: usize,
}

impl FlatGraph {
    /// Build: exact brute-force local k-NN + random long-jump edges.
    pub fn build(vectors: Vec<f32>, config: GraphConfig) -> Self {
        let n = vectors.len() / config.dims;
        assert_eq!(vectors.len(), n * config.dims, "vector length mismatch");
        let m = config.m.min(n.saturating_sub(1));
        let dims = config.dims;

        // Local k-NN (parallel)
        let mut neighbors: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vi = &vectors[i * dims..(i + 1) * dims];
                let mut dists: Vec<(u32, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let vj = &vectors[j * dims..(j + 1) * dims];
                        (j as u32, l2_sq(vi, vj))
                    })
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                dists.truncate(m);
                dists.into_iter().map(|(idx, _)| idx).collect()
            })
            .collect();

        // Long-jump edges (single-threaded for determinism)
        let m_lj = config.m_longjump.min(n.saturating_sub(1));
        if m_lj > 0 {
            let mut rng = StdRng::seed_from_u64(0xA17E_BABEu64);
            let pick = Uniform::new(0usize, n);
            for i in 0..n {
                let mut added = 0;
                let mut tries = 0;
                while added < m_lj && tries < m_lj * 8 {
                    tries += 1;
                    let j = pick.sample(&mut rng);
                    if j != i && !neighbors[i].contains(&(j as u32)) {
                        neighbors[i].push(j as u32);
                        added += 1;
                    }
                }
            }
        }

        FlatGraph {
            vectors,
            neighbors,
            config: GraphConfig { m, m_longjump: m_lj, dims },
            n,
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let d = self.config.dims;
        &self.vectors[i * d..(i + 1) * d]
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Squared L2 — no sqrt; consistent with HNSW conventions.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_connects_all_nodes() {
        let v: Vec<f32> = (0..40).map(|i| i as f32 / 40.0).collect();
        let g = FlatGraph::build(
            v,
            GraphConfig { m: 3, m_longjump: 2, dims: 4 },
        );
        assert_eq!(g.n, 10);
        for (i, nb) in g.neighbors.iter().enumerate() {
            assert!(!nb.is_empty());
            for &x in nb {
                assert_ne!(x as usize, i);
                assert!((x as usize) < g.n);
            }
        }
    }
}
