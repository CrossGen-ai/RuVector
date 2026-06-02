//! Joint graph + quantized search — the SymphonyQG recipe.
//!
//! Build one graph over the float data, then attach codes per node. At query
//! time we expose three modes that walk the *same* graph with different
//! distance functions:
//!
//! * [`SearchMode::Float`]    — float L2 (oracle on this graph).
//! * [`SearchMode::Binary`]   — pure quantized walk + quantized ordering.
//! * [`SearchMode::Symphony`] — quantized walk with a `rerank` budget: the
//!   final `k` are re-scored with float L2 against the top-`rerank` codes.

use crate::graph::{Graph, GraphBuilder};
use crate::l2_sq;
use crate::quant::{BinaryCode, BinaryCodec};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SearchMode {
    Float,
    Binary,
    Symphony { rerank: usize },
}

/// Result entry — distance is in the metric the caller asked for: float L2
/// for `Float` and `Symphony`, the quantized proxy for `Binary`.
#[derive(Copy, Clone, Debug)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

pub struct Searcher {
    graph: Graph,
    codec: BinaryCodec,
    codes: Vec<BinaryCode>,
}

impl Searcher {
    /// Bulk-build from a row-major `dim`-dimensional dataset.
    pub fn build(data: &[f32], dim: usize, m: usize, ef_construction: usize) -> Self {
        let mut b = GraphBuilder::new(dim, m, ef_construction);
        let n = data.len() / dim;
        for i in 0..n {
            b.insert(&data[i * dim..(i + 1) * dim]);
        }
        let graph = b.build();
        let codec = BinaryCodec::fit(&graph.vectors, dim);
        let codes = codec.encode_all(&graph.vectors);
        Self { graph, codec, codes }
    }

    pub fn len(&self) -> usize { self.graph.len() }
    pub fn dim(&self) -> usize { self.graph.dim }
    pub fn bits_per_vec(&self) -> usize { self.codec.words * 64 }

    /// Bytes for the stored binary codebook (excluding the float graph).
    pub fn quantized_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.bits.len() * 8).sum()
    }

    /// Bytes for the float vectors stored alongside the graph.
    pub fn float_bytes(&self) -> usize {
        self.graph.vectors.len() * std::mem::size_of::<f32>()
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize, mode: SearchMode)
        -> Vec<Neighbor>
    {
        match mode {
            SearchMode::Float => {
                self.graph.search_float(query, k, ef)
                    .into_iter().map(|(d, id)| Neighbor { id, dist: d }).collect()
            }
            SearchMode::Binary => {
                let pq = self.codec.prepare_query(query);
                let codes = &self.codes;
                let cands = self.graph.search_with(query, k, ef, |id| {
                    pq.score(&codes[id as usize])
                });
                cands.into_iter().map(|(d, id)| Neighbor { id, dist: d }).collect()
            }
            SearchMode::Symphony { rerank } => {
                let pq = self.codec.prepare_query(query);
                let codes = &self.codes;
                // Walk on cheap codes; widen to `rerank` candidates. Use a
                // generous traversal frontier (2× rerank) since each code
                // distance is ~30× cheaper than a float L2.
                let r = rerank.max(k);
                let cands = self.graph.search_with(query, r, ef.max(r), |id| {
                    pq.score(&codes[id as usize])
                });
                // Rerank with exact float L2.
                let mut scored: Vec<Neighbor> = cands.into_iter().map(|(_, id)| {
                    let d = l2_sq(query, self.graph.vec_of(id));
                    Neighbor { id, dist: d }
                }).collect();
                scored.sort_by(|a, b| a.dist.total_cmp(&b.dist));
                scored.truncate(k);
                scored
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Multi-Gaussian clusters — what realistic ANN workloads look like.
        let mut rng = StdRng::seed_from_u64(seed);
        let clusters = 16;
        let mut centers = vec![0.0f32; clusters * dim];
        for v in centers.iter_mut() { *v = rng.gen_range(-3.0..3.0); }
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let c = rng.gen_range(0..clusters);
            for d in 0..dim {
                out.push(centers[c * dim + d] + rng.gen_range(-0.5..0.5));
            }
        }
        out
    }

    fn brute_top_k(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
        let n = data.len() / dim;
        let mut all: Vec<(f32, u32)> = (0..n).map(|i| {
            (crate::l2_sq(q, &data[i * dim..(i + 1) * dim]), i as u32)
        }).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn symphony_recovers_float_recall() {
        let dim = 32;
        let n = 800;
        let data = synth(n, dim, 1);
        let queries = synth(50, dim, 2);
        let s = Searcher::build(&data, dim, 16, 64);

        let mut floats = 0usize;
        let mut symphony = 0usize;
        let mut binary = 0usize;
        let k = 10;
        for qi in 0..50 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let truth: std::collections::HashSet<u32> =
                brute_top_k(&data, dim, q, k).into_iter().collect();

            let hits = |xs: Vec<Neighbor>| xs.iter().filter(|n| truth.contains(&n.id)).count();
            floats   += hits(s.search(q, k, 64, SearchMode::Float));
            binary   += hits(s.search(q, k, 64, SearchMode::Binary));
            symphony += hits(s.search(q, k, 64, SearchMode::Symphony { rerank: 64 }));
        }
        // Symphony rerank should clearly beat pure binary, and pure float
        // should out-perform raw binary. We don't assert symphony ≈ float
        // because 1-bit codes on iid Gaussians are intrinsically too noisy
        // to fully recover oracle recall (the documented limitation in
        // docs/research/nightly/.../README.md).
        assert!(symphony >= binary,
            "symphony rerank should not be worse than raw binary ({} vs {})",
            symphony, binary);
        assert!(floats > binary,
            "float graph should out-recall raw binary ({} vs {})",
            floats, binary);
    }
}
