//! Baseline IVF-Flat: each corpus vector is assigned to exactly one centroid,
//! the nearest one. This is the reference against which SOAR and top-k spilling
//! are compared.

use crate::ivf_common::{
    centroid_bytes, nearest_centroid_idx, posting_bytes, probe_and_scan, probe_and_scan_approx,
    Centroids, IvfVariant, PostingLists,
};
use crate::metrics::Hit;
use crate::sq_l2;

pub struct IvfSingle<'v> {
    pub centroids: Centroids,
    pub lists: PostingLists,
    pub vectors: &'v [Vec<f32>],
}

impl<'v> IvfSingle<'v> {
    /// Build the index. `n_centroids` and `iters` control the k-means-lite
    /// centroid trainer.
    pub fn build(vectors: &'v [Vec<f32>], n_centroids: usize, iters: usize, seed: u64) -> Self {
        let centroids = Centroids::train(vectors, n_centroids, iters, seed);
        let mut lists = PostingLists::new(centroids.len());
        for (id, v) in vectors.iter().enumerate() {
            let c = nearest_centroid_idx(v, &centroids.vecs);
            let rn = sq_l2(v, &centroids.vecs[c]).sqrt();
            lists.push_with_norm(c, id, rn);
        }
        Self {
            centroids,
            lists,
            vectors,
        }
    }
}

impl<'v> IvfVariant for IvfSingle<'v> {
    fn name(&self) -> &str {
        "IVF-Single"
    }

    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<Hit> {
        probe_and_scan(query, k, n_probe, &self.centroids, &self.lists, self.vectors)
    }

    fn search_approx(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<Hit> {
        probe_and_scan_approx(query, k, n_probe, &self.centroids, &self.lists)
    }

    fn memory_bytes(&self) -> usize {
        centroid_bytes(&self.centroids) + posting_bytes(&self.lists)
    }

    fn spill_overhead(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, DatasetConfig};

    #[test]
    fn total_postings_equal_n_vectors() {
        let cfg = DatasetConfig {
            n_vectors: 500,
            dims: 8,
            n_queries: 10,
            n_clusters: 8,
            sigma: 0.5,
            center_sigma: 5.0,
            seed: 7,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSingle::build(&d.vectors, 16, 8, 42);
        assert_eq!(idx.lists.total_entries, 500);
        assert_eq!(idx.spill_overhead(), 0);
    }

    #[test]
    fn search_returns_k_or_fewer_hits() {
        let cfg = DatasetConfig {
            n_vectors: 200,
            dims: 8,
            n_queries: 5,
            n_clusters: 4,
            sigma: 0.5,
            center_sigma: 4.0,
            seed: 11,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSingle::build(&d.vectors, 8, 6, 42);
        let hits = idx.search(&d.queries[0], 10, 4);
        assert!(hits.len() <= 10);
        // Distances should be ascending.
        for w in hits.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }
}
