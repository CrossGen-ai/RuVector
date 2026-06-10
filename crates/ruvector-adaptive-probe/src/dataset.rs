//! Dataset helpers reused by the demo binary and the criterion bench.

use crate::make_synthetic;

pub struct Workload {
    pub corpus: Vec<f32>,
    pub queries: Vec<f32>,
    pub n: usize,
    pub n_queries: usize,
    pub dim: usize,
}

impl Workload {
    pub fn gaussian(n: usize, n_queries: usize, dim: usize, n_clusters: usize, seed: u64) -> Self {
        // Corpus and queries drawn from the same generative process but with
        // different seeds so queries are out-of-sample but in-distribution.
        let corpus = make_synthetic(n, dim, n_clusters, 0.3, seed);
        let queries = make_synthetic(n_queries, dim, n_clusters, 0.3, seed.wrapping_add(0xdead));
        Self {
            corpus,
            queries,
            n,
            n_queries,
            dim,
        }
    }

    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }
}
