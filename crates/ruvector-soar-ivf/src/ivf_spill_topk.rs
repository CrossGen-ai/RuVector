//! Top-k spilling baseline: each corpus vector is duplicated into its `s`
//! nearest centroids. This is the "naive" way to get partition redundancy —
//! SOAR should match its recall at *smaller* spillover, or beat it at equal
//! spillover, because SOAR picks the secondary by residual complementarity
//! rather than by residual magnitude alone.

use crate::ivf_common::{
    centroid_bytes, posting_bytes, probe_and_scan, probe_and_scan_approx, rank_centroids,
    Centroids, IvfVariant, PostingLists,
};
use crate::metrics::Hit;

pub struct IvfSpillTopK<'v> {
    pub centroids: Centroids,
    pub lists: PostingLists,
    pub vectors: &'v [Vec<f32>],
    pub spill: usize,
    pub duplicates: usize,
}

impl<'v> IvfSpillTopK<'v> {
    /// `spill` = number of centroids each vector is copied into (>=1). spill=1
    /// degenerates to `IvfSingle`.
    pub fn build(
        vectors: &'v [Vec<f32>],
        n_centroids: usize,
        spill: usize,
        iters: usize,
        seed: u64,
    ) -> Self {
        assert!(spill >= 1);
        let centroids = Centroids::train(vectors, n_centroids, iters, seed);
        let mut lists = PostingLists::new(centroids.len());
        let mut dup = 0usize;
        for (id, v) in vectors.iter().enumerate() {
            let ranked = rank_centroids(v, &centroids.vecs);
            for (rank, (cid, dsq)) in ranked.into_iter().take(spill).enumerate() {
                lists.push_with_norm(cid, id, dsq.sqrt());
                if rank >= 1 {
                    dup += 1;
                }
            }
        }
        lists.dup_count = dup;
        Self {
            centroids,
            lists,
            vectors,
            spill,
            duplicates: dup,
        }
    }
}

impl<'v> IvfVariant for IvfSpillTopK<'v> {
    fn name(&self) -> &str {
        "IVF-SpillTopK"
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
        self.duplicates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, DatasetConfig};

    #[test]
    fn spill_factor_multiplies_postings() {
        let cfg = DatasetConfig {
            n_vectors: 200,
            dims: 8,
            n_queries: 5,
            n_clusters: 4,
            sigma: 0.5,
            center_sigma: 4.0,
            seed: 3,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSpillTopK::build(&d.vectors, 16, 2, 6, 42);
        assert_eq!(idx.lists.total_entries, 200 * 2);
        assert_eq!(idx.spill_overhead(), 200);
    }

    #[test]
    fn spill_one_matches_single() {
        // spill=1 → duplicate count 0.
        let cfg = DatasetConfig {
            n_vectors: 100,
            dims: 6,
            n_queries: 2,
            n_clusters: 3,
            sigma: 0.3,
            center_sigma: 3.0,
            seed: 5,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSpillTopK::build(&d.vectors, 8, 1, 4, 42);
        assert_eq!(idx.spill_overhead(), 0);
        assert_eq!(idx.lists.total_entries, 100);
    }
}
