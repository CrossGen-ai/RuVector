//! ruvector-tribase: Triangle-inequality pivot pruning for IVF vector search.
//!
//! Three search backends share a single dataset abstraction:
//!   * [`flat::FlatIndex`] — exhaustive brute-force baseline (recall = 1.0).
//!   * [`ivf::IvfIndex`]   — standard IVF, scans every vector in nprobe lists.
//!   * [`tribase::TribaseIndex`] — IVF + triangle-inequality LB/UB pruning.
//!
//! All three return identical results when nprobe is large enough (Tribase is
//! exact within a probed list: it only skips vectors that *cannot* be top-k).

pub mod dist;
pub mod flat;
pub mod ivf;
pub mod kmeans;
pub mod tribase;

pub use dist::{sq_l2, TopK};
pub use flat::{FlatIndex, SearchStats};
pub use ivf::IvfIndex;
pub use tribase::TribaseIndex;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    /// Clustered synthetic data: nc Gaussian clusters with small noise so
    /// triangle bounds are meaningful (pruning relies on tight tau).
    fn synth_clustered(n: usize, dim: usize, nc: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..nc)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect())
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % nc];
                c.iter()
                    .map(|&v| v + (rng.gen::<f32>() * 2.0 - 1.0) * sigma)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn flat_top_k_correct() {
        let data = synth(200, 16, 1);
        let idx = FlatIndex::build(data.clone());
        let q = data[0].clone();
        let (r, _) = idx.search(&q, 5);
        assert_eq!(r.len(), 5);
        // Self-match should be top-1 with distance 0
        assert_eq!(r[0].1, 0);
        assert!(r[0].0 < 1e-6);
    }

    #[test]
    fn tribase_matches_ivf_when_fully_probed() {
        let data = synth(500, 12, 7);
        let ivf = IvfIndex::build(data.clone(), 16, 8, 7);
        let tri = TribaseIndex::build(data.clone(), 16, 8, 7);
        // Same k-means seed -> same partition. Fully probe to remove
        // ordering effects.
        let q = synth(1, 12, 99).remove(0);
        let (r_ivf, _) = ivf.search(&q, 10, 16);
        let (r_tri, _) = tri.search(&q, 10, 16);
        assert_eq!(r_ivf.len(), r_tri.len());
        for i in 0..r_ivf.len() {
            // Distances must be identical (Tribase is exact within probed lists)
            assert!((r_ivf[i].0 - r_tri[i].0).abs() < 1e-5,
                "rank {}: ivf={} tri={}", i, r_ivf[i].0, r_tri[i].0);
        }
    }

    #[test]
    fn tribase_prunes() {
        // Clustered data has structure -> tight tau -> pruning triggers
        let data = synth_clustered(2000, 32, 16, 0.3, 3);
        let ivf = IvfIndex::build(data.clone(), 32, 15, 3);
        let tri = TribaseIndex::build(data.clone(), 32, 15, 3);
        let q = data[42].clone();
        let (_, s_ivf) = ivf.search(&q, 10, 8);
        let (_, s_tri) = tri.search(&q, 10, 8);
        // Tribase must do strictly fewer distance computations on the data
        // (we don't count centroid distances in this assertion since both
        // pay that cost identically).
        let ivf_data_dists = s_ivf.dist_computations - 32; // minus centroid pass
        let tri_data_dists = s_tri.dist_computations - 32;
        assert!(
            tri_data_dists < ivf_data_dists,
            "tribase did not prune: ivf={} tri={}",
            ivf_data_dists, tri_data_dists
        );
    }

    #[test]
    fn topk_heap_correct() {
        let mut h = TopK::new(3);
        for v in [5.0_f32, 1.0, 4.0, 2.0, 3.0] {
            h.push(v, 0);
        }
        let out: Vec<f32> = h.into_sorted().into_iter().map(|(d, _)| d).collect();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }
}
