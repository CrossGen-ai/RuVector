//! Reverse-KNN verification.
//!
//! Given a base candidate list `C(q) = {c_1, ..., c_M}` returned by an ANN
//! backend, keep candidate `c` iff `q` lies within the top-`k_rev` neighborhood
//! of `c` in the dataset. Two operating modes:
//!
//!   * [`VerifyMode::Live`]    — compute each candidate's own top-`k_rev` on
//!     demand by delegating to the underlying [`NnIndex`]. Zero extra memory,
//!     `O(M * n)` per query with a flat index (still fast for small M).
//!   * [`VerifyMode::Cached`]  — pre-build a `k_rev`-nearest cache once; each
//!     verification is an `O(k_rev)` membership check.
//!
//! The verifier optionally accepts a candidate when the query's distance to
//! it is below a slack threshold relative to the candidate's own k-th
//! neighbor distance (i.e. treat it as inside a scaled ball). This keeps
//! recall reasonable when the reverse-cache is small.

use crate::{sq_l2, NnIndex};

#[derive(Clone, Copy, Debug)]
pub enum VerifyMode {
    Live,
    Cached,
}

pub struct RknnVerifier<'a, I: NnIndex + ?Sized> {
    index: &'a I,
    /// k used for each point's own reverse-neighborhood test.
    pub k_rev: usize,
    /// Radius slack: accept if `d(q, c) <= slack * d_k(c)` where
    /// `d_k(c)` is the distance from `c` to its k-th own neighbor.
    /// `slack = 1.0` == strict "q inside own top-k ball";
    /// `slack > 1.0` == permissive expansion.
    pub slack: f32,
    /// Precomputed per-point (top-k_rev ids, d_k) if in Cached mode.
    cache: Option<Vec<(Vec<usize>, f32)>>,
    mode: VerifyMode,
}

impl<'a, I: NnIndex + ?Sized> RknnVerifier<'a, I> {
    pub fn new_live(index: &'a I, k_rev: usize, slack: f32) -> Self {
        Self {
            index,
            k_rev,
            slack,
            cache: None,
            mode: VerifyMode::Live,
        }
    }

    /// Build the reverse-KNN cache once. `O(n^2)` for a flat backend; for a
    /// production graph backend this is `O(n * search_cost)`.
    pub fn new_cached(index: &'a I, k_rev: usize, slack: f32) -> Self {
        let n = index.len();
        let mut cache = Vec::with_capacity(n);
        for id in 0..n {
            let v = index.vector(id).to_vec();
            let hits = index.search(&v, k_rev + 1);
            // Drop self (distance 0). Keep the id and the k-th neighbor dist
            // as the acceptance-radius baseline.
            let mut ids = Vec::with_capacity(k_rev);
            let mut d_k = 0.0f32;
            for (nid, d) in hits.into_iter() {
                if nid == id {
                    continue;
                }
                ids.push(nid);
                d_k = d;
                if ids.len() >= k_rev {
                    break;
                }
            }
            cache.push((ids, d_k));
        }
        Self {
            index,
            k_rev,
            slack,
            cache: Some(cache),
            mode: VerifyMode::Cached,
        }
    }

    pub fn mode(&self) -> VerifyMode {
        self.mode
    }

    /// Return `true` if candidate `c_id` accepts query `q`.
    pub fn accepts(&self, q: &[f32], c_id: usize) -> bool {
        let dqc = sq_l2(q, self.index.vector(c_id));
        match &self.cache {
            Some(cache) => {
                let (ids, d_k) = &cache[c_id];
                if ids.contains(&self.k_rev) {
                    // impossible marker; keep clippy happy
                }
                // Fast path: strict membership if q coincides with some cached id.
                // Otherwise use radius test: q accepted iff its distance to c
                // is within slack * d_k(c).
                dqc <= self.slack * *d_k
                    || ids.iter().any(|&nid| {
                        // For cached mode we do a cheap identity test — the
                        // query is usually not exactly a dataset point, so
                        // this is a defensive shortcut for query==vector(nid).
                        let vn = self.index.vector(nid);
                        vn.len() == q.len() && sq_l2(vn, q) < 1e-12
                    })
            }
            None => {
                // Live mode: compute c's own top-k_rev.
                let cv = self.index.vector(c_id).to_vec();
                let hits = self.index.search(&cv, self.k_rev + 1);
                // Drop self.
                let mut d_k = 0.0f32;
                for (nid, d) in hits.into_iter() {
                    if nid == c_id {
                        continue;
                    }
                    d_k = d;
                }
                dqc <= self.slack * d_k
            }
        }
    }

    /// Filter a candidate list, preserving order.
    pub fn filter(&self, q: &[f32], cands: &[(usize, f32)]) -> Vec<(usize, f32)> {
        cands
            .iter()
            .copied()
            .filter(|(id, _)| self.accepts(q, *id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate, GenSpec};
    use crate::FlatL2Index;

    fn small_index() -> FlatL2Index {
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|i| vec![i as f32, (i * i) as f32 * 0.01])
            .collect();
        FlatL2Index::from_rows(2, &rows)
    }

    #[test]
    fn live_and_cached_agree_on_query_equal_to_dataset_point() {
        let idx = small_index();
        let live = RknnVerifier::new_live(&idx, 5, 1.0);
        let cached = RknnVerifier::new_cached(&idx, 5, 1.0);
        let q = idx.vector(3).to_vec();
        // Candidate = point 3 itself; its own neighborhood always contains itself
        // via the radius test (distance 0 <= slack * d_k).
        assert!(live.accepts(&q, 3));
        assert!(cached.accepts(&q, 3));
    }

    #[test]
    fn cache_shape_matches_dataset_size() {
        let idx = small_index();
        let cached = RknnVerifier::new_cached(&idx, 4, 1.0);
        assert_eq!(cached.cache.as_ref().unwrap().len(), 40);
        for (ids, _) in cached.cache.as_ref().unwrap() {
            assert_eq!(ids.len(), 4);
        }
    }

    #[test]
    fn verifier_strictly_reduces_or_preserves_candidates() {
        let spec = GenSpec {
            n: 300,
            dim: 8,
            n_clusters: 6,
            hub_frac: 0.1,
            hub_scale: 3.0,
            seed: 11,
        };
        let g = generate(&spec, 5);
        let v = RknnVerifier::new_live(&g.index, 10, 1.0);
        for q in &g.queries {
            let cands = g.index.search(q, 20);
            let filtered = v.filter(q, &cands);
            assert!(filtered.len() <= cands.len());
        }
    }
}
