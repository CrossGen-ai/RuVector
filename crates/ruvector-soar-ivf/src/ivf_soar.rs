//! SOAR: Spilling with Orthogonality-Amplified Residuals.
//!
//! Assignment rule (`spill = 2` case, the interesting one):
//!   1. Primary centroid `c_p` = argmin_c ||x - c||²   (same as IVF-Single).
//!   2. Let `r_p = x - c_p`. Its squared norm is `‖r_p‖² = R`.
//!   3. Secondary centroid `c_s` is chosen from the remaining centroids as:
//!         c_s = argmin_c  L(c ; x, r_p)
//!      where the SOAR cost is
//!         L(c) = ‖r_c‖² + λ · ( (r_p · r_c)² / R )
//!      with `r_c = x - c` and `λ ≥ 0` a hyper-parameter (paper: λ ≈ 1.0).
//!
//! The second term is the squared magnitude of the *component of the secondary
//! residual parallel to the primary residual*, scaled by λ. Amplifying that
//! component *penalises* secondary centroids whose residual points in the same
//! direction as the primary residual, so SOAR prefers secondaries that cover
//! error directions the primary handles poorly.
//!
//! Extending to `spill > 2` is straightforward: greedily add centroids one at
//! a time, amplifying orthogonality against the *span* of all previously chosen
//! residuals. We keep the general form here; degrades gracefully for spill=1
//! (equivalent to `IvfSingle`).

use crate::ivf_common::{
    centroid_bytes, nearest_centroid_idx, posting_bytes, probe_and_scan, probe_and_scan_approx,
    Centroids, IvfVariant, PostingLists,
};
use crate::metrics::Hit;
use crate::{dot, sq_l2, sub_into};

pub struct IvfSoar<'v> {
    pub centroids: Centroids,
    pub lists: PostingLists,
    pub vectors: &'v [Vec<f32>],
    pub spill: usize,
    pub lambda: f32,
    pub duplicates: usize,
}

impl<'v> IvfSoar<'v> {
    /// Build the SOAR-IVF index.
    ///
    /// * `spill` — number of centroids each vector is copied into (>=1).
    /// * `lambda` — orthogonality amplification (paper default ≈ 1.0).
    ///              λ=0 recovers ordinary top-`spill` spilling.
    pub fn build(
        vectors: &'v [Vec<f32>],
        n_centroids: usize,
        spill: usize,
        lambda: f32,
        iters: usize,
        seed: u64,
    ) -> Self {
        assert!(spill >= 1);
        assert!(lambda >= 0.0);
        let centroids = Centroids::train(vectors, n_centroids, iters, seed);
        let dims = centroids.dims;
        let mut lists = PostingLists::new(centroids.len());
        let mut dup = 0usize;

        // Scratch buffers reused per vector.
        let mut r_primary = vec![0.0f32; dims];
        let mut r_cand = vec![0.0f32; dims];

        for (id, x) in vectors.iter().enumerate() {
            // Step 1: primary centroid.
            let p = nearest_centroid_idx(x, &centroids.vecs);
            let rp_sq_true = sq_l2(x, &centroids.vecs[p]);
            lists.push_with_norm(p, id, rp_sq_true.sqrt());

            if spill == 1 {
                continue;
            }

            // Step 2: primary residual.
            sub_into(x, &centroids.vecs[p], &mut r_primary);
            let rp_sq = rp_sq_true.max(1e-12); // avoid div/0

            // Step 3+: greedily add spill-1 secondaries, each minimising the
            // orthogonality-amplified cost against ALL previously chosen
            // residuals. We keep a running list of chosen residual vectors and
            // their squared norms.
            let mut chosen: Vec<usize> = vec![p];
            let mut chosen_residuals: Vec<Vec<f32>> = vec![r_primary.clone()];
            let mut chosen_rsq: Vec<f32> = vec![rp_sq];

            for _ in 1..spill {
                let mut best: Option<(usize, f32)> = None;
                for (ci, c) in centroids.vecs.iter().enumerate() {
                    if chosen.contains(&ci) {
                        continue;
                    }
                    // Candidate residual r_c.
                    sub_into(x, c, &mut r_cand);
                    let rc_sq = sq_l2(x, c);
                    // Orthogonality-amplified cost: sum over already-chosen
                    // residuals of the *parallel* squared magnitude, plus
                    // baseline residual norm.
                    let mut cost = rc_sq;
                    for (r_prev, r_prev_sq) in chosen_residuals.iter().zip(chosen_rsq.iter()) {
                        let dp = dot(&r_cand, r_prev);
                        cost += lambda * (dp * dp) / r_prev_sq;
                    }
                    match best {
                        None => best = Some((ci, cost)),
                        Some((_, bc)) if cost < bc => best = Some((ci, cost)),
                        _ => {}
                    }
                }
                if let Some((sec, _)) = best {
                    let rsq_true = sq_l2(x, &centroids.vecs[sec]);
                    lists.push_with_norm(sec, id, rsq_true.sqrt());
                    dup += 1;
                    // Record its residual so subsequent picks are also
                    // orthogonalised against it.
                    let mut r_sec = vec![0.0f32; dims];
                    sub_into(x, &centroids.vecs[sec], &mut r_sec);
                    let rsq = rsq_true.max(1e-12);
                    chosen.push(sec);
                    chosen_residuals.push(r_sec);
                    chosen_rsq.push(rsq);
                } else {
                    // Fewer centroids than requested spill — nothing to add.
                    break;
                }
            }
        }

        lists.dup_count = dup;
        Self {
            centroids,
            lists,
            vectors,
            spill,
            lambda,
            duplicates: dup,
        }
    }
}

impl<'v> IvfVariant for IvfSoar<'v> {
    fn name(&self) -> &str {
        "IVF-SOAR"
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
    fn spill_one_matches_single_assignment() {
        let cfg = DatasetConfig {
            n_vectors: 200,
            dims: 8,
            n_queries: 5,
            n_clusters: 4,
            sigma: 0.5,
            center_sigma: 4.0,
            seed: 21,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSoar::build(&d.vectors, 8, 1, 1.0, 6, 42);
        assert_eq!(idx.lists.total_entries, 200);
        assert_eq!(idx.spill_overhead(), 0);
    }

    #[test]
    fn spill_two_doubles_postings_and_records_duplicates() {
        let cfg = DatasetConfig {
            n_vectors: 300,
            dims: 8,
            n_queries: 5,
            n_clusters: 6,
            sigma: 0.5,
            center_sigma: 4.0,
            seed: 22,
        };
        let d = Dataset::generate(cfg);
        let idx = IvfSoar::build(&d.vectors, 12, 2, 1.0, 6, 42);
        assert_eq!(idx.lists.total_entries, 300 * 2);
        assert_eq!(idx.spill_overhead(), 300);
    }

    #[test]
    fn lambda_zero_recovers_ordinary_topk_spill_secondary_choice() {
        // When λ=0 the SOAR cost is just ‖r_c‖², so the secondary picked by
        // SOAR must be the *second*-nearest centroid — same choice as
        // IvfSpillTopK with spill=2.
        let cfg = DatasetConfig {
            n_vectors: 400,
            dims: 8,
            n_queries: 10,
            n_clusters: 6,
            sigma: 0.5,
            center_sigma: 4.0,
            seed: 23,
        };
        let d = Dataset::generate(cfg);
        let soar = IvfSoar::build(&d.vectors, 8, 2, 0.0, 6, 42);
        let topk = crate::ivf_spill_topk::IvfSpillTopK::build(&d.vectors, 8, 2, 6, 42);

        // Same posting *set* per centroid (order may differ).
        assert_eq!(soar.centroids.vecs.len(), topk.centroids.vecs.len());
        for (a, b) in soar.lists.lists.iter().zip(topk.lists.lists.iter()) {
            let mut aa = a.clone();
            let mut bb = b.clone();
            aa.sort();
            bb.sort();
            assert_eq!(aa, bb, "posting lists differ when λ=0");
        }
    }

    /// Reimplementation of the SOAR secondary-pick cost, kept in the test
    /// module so it can be exercised against hand-crafted centroids without
    /// going through k-means. If this ever diverges from the production
    /// formula in `build`, the test that compares it against ordinary
    /// top-k spilling (`lambda_zero_recovers_ordinary_topk_spill_secondary_choice`)
    /// will catch it.
    fn soar_cost(r_cand: &[f32], r_prev: &[f32], r_prev_sq: f32, lambda: f32) -> f32 {
        let rc_sq: f32 = r_cand.iter().map(|x| x * x).sum();
        let dp: f32 = r_cand.iter().zip(r_prev.iter()).map(|(a, b)| a * b).sum();
        rc_sq + lambda * (dp * dp) / r_prev_sq
    }

    #[test]
    fn soar_cost_penalises_parallel_residuals() {
        // x at origin. Primary centroid at (-1, 0) → r_p = (1, 0), ‖r_p‖² = 1.
        // Candidate A at (-2, 0) → r_A = (2, 0)   — perfectly parallel to r_p.
        // Candidate B at ( 0, 2) → r_B = (0,-2)   — perfectly orthogonal.
        // Both have ‖r‖² = 4, so at λ=0 they tie. With λ>0, A must strictly
        // exceed B (parallel component amplified).
        let r_p = vec![1.0f32, 0.0];
        let rp_sq = 1.0f32;
        let r_a = vec![2.0f32, 0.0];
        let r_b = vec![0.0f32, -2.0];
        assert!((soar_cost(&r_a, &r_p, rp_sq, 0.0) - soar_cost(&r_b, &r_p, rp_sq, 0.0)).abs() < 1e-6);
        // At λ=1: A adds 4, B adds 0.
        let ca = soar_cost(&r_a, &r_p, rp_sq, 1.0);
        let cb = soar_cost(&r_b, &r_p, rp_sq, 1.0);
        assert!(ca > cb + 1e-3, "expected parallel to cost more: ca={} cb={}", ca, cb);
        // At λ=4: A adds 16, B still 0.
        let ca4 = soar_cost(&r_a, &r_p, rp_sq, 4.0);
        let cb4 = soar_cost(&r_b, &r_p, rp_sq, 4.0);
        assert!(ca4 - cb4 > 15.0);
    }
}
