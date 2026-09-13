//! Precomputed corpus-wide hubness statistics used by all rerankers.
//!
//! For each corpus vector `v`, we store:
//!   * `sorted_dists[v]`  — the sorted distances from `v` to every other
//!     corpus vector. Used by Mutual Proximity to evaluate the empirical CDF.
//!   * `kth_dist[v]`      — the k-th nearest-neighbour distance of `v` (i.e.
//!     `sorted_dists[v][k]`). Used by Local Scaling.
//!   * `topk_mean[v]`     — the arithmetic mean of the top-k distances of
//!     `v`. Used by NICDM.
//!
//! Memory: `O(N^2)` for `sorted_dists`. That is fine for the PoC (N=5000
//! ~= 100 MB with f32) and matches the memory profile of skhubness's
//! reference implementation. In production, a sampled or LSH-approximated
//! CDF gets us to `O(N·s)` with `s≈2048` and negligible quality loss — see
//! the roadmap in the research doc.

use crate::distance::Distance;
use crate::index::Vector;

/// Corpus-wide hubness statistics needed by the rerankers.
pub struct HubnessStats {
    /// `sorted_dists[i]` = distances from corpus[i] to every other corpus
    /// point (including itself as 0.0 at position 0), sorted ascending.
    pub sorted_dists: Vec<Vec<f32>>,
    /// k-NN distance of every corpus vector (used by Local Scaling).
    pub kth_dist: Vec<f32>,
    /// Arithmetic mean of the top-k NN distances (used by NICDM).
    pub topk_mean: Vec<f32>,
    /// The `k` used to compute `kth_dist` / `topk_mean`.
    pub k: usize,
}

impl HubnessStats {
    /// N of the corpus.
    pub fn len(&self) -> usize {
        self.sorted_dists.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sorted_dists.is_empty()
    }
}

/// Compute the hubness statistics for the given corpus under `distance`.
/// `k` is the neighbourhood size used by Local Scaling / NICDM.
///
/// Cost: O(N^2·D) distance work + O(N^2·log N) sorting.
pub fn hubness_stats<D: Distance>(corpus: &[Vector], distance: &D, k: usize) -> HubnessStats {
    let n = corpus.len();
    assert!(k >= 1, "k must be ≥ 1");
    assert!(k < n, "k must be < corpus size");

    // Compute the full symmetric distance matrix in row form. We store each
    // row separately so downstream code can sort in-place without touching
    // its neighbours' data.
    let mut sorted_dists = Vec::with_capacity(n);
    let mut dist_row = vec![0f32; n];
    for i in 0..n {
        for j in 0..n {
            // The empirical CDF cares about the *distribution* of distances
            // from i to everything else; the self-distance sits at 0 and
            // does not perturb tail probabilities used at query time (a
            // query is not a corpus point).
            dist_row[j] = if i == j {
                0.0
            } else {
                distance.dist(&corpus[i], &corpus[j])
            };
        }
        let mut row = dist_row.clone();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_dists.push(row);
    }

    // kth distance and mean of top-k. Skip position 0 which is the self-
    // distance (0.0). Top-k window is therefore [1..=k].
    let mut kth_dist = Vec::with_capacity(n);
    let mut topk_mean = Vec::with_capacity(n);
    for i in 0..n {
        let row = &sorted_dists[i];
        kth_dist.push(row[k]); // 1-indexed skip of self
        let sum: f32 = row[1..=k].iter().sum();
        topk_mean.push(sum / k as f32);
    }

    HubnessStats {
        sorted_dists,
        kth_dist,
        topk_mean,
        k,
    }
}

/// Empirical survival function: `P(D_v > d) = |{x : sorted_dists_v[x] > d}| / N`.
///
/// Uses binary search on the sorted row. The +1/N Laplace smoothing avoids
/// zero-probability tails that would zero-out Mutual Proximity scores for
/// the very closest neighbours.
#[inline]
pub fn empirical_survival(sorted_row: &[f32], d: f32) -> f32 {
    // Number of elements strictly greater than d.
    let n = sorted_row.len();
    // partition_point returns the first index where predicate is false,
    // i.e. the first index whose value is > d. Everything from there to
    // the end is > d.
    let leq = sorted_row.partition_point(|x| *x <= d);
    let gt = n - leq;
    // Laplace: (gt + 1) / (n + 2). Never zero, never one.
    (gt as f32 + 1.0) / (n as f32 + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::L2;

    fn tiny_corpus() -> Vec<Vector> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![10.0, 0.0],
        ]
    }

    #[test]
    fn stats_shapes_match_corpus() {
        let c = tiny_corpus();
        let s = hubness_stats(&c, &L2, 2);
        assert_eq!(s.len(), 4);
        assert_eq!(s.kth_dist.len(), 4);
        assert_eq!(s.topk_mean.len(), 4);
        // First point's nearest is at distance 1, second-nearest at 2.
        assert!((s.kth_dist[0] - 2.0).abs() < 1e-6);
        assert!((s.topk_mean[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn survival_is_monotone_decreasing() {
        let row = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let s0 = empirical_survival(&row, 0.0);
        let s2 = empirical_survival(&row, 2.0);
        let s5 = empirical_survival(&row, 5.0);
        assert!(s0 > s2);
        assert!(s2 > s5);
        // Never zero, never one — Laplace smoothing.
        assert!(s5 > 0.0);
        assert!(s0 < 1.0);
    }
}
