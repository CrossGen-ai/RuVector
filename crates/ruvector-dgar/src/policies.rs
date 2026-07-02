//! Rerank policies: FixedK, AdaptiveGap (DGAR), OracleUpperBound.
//!
//! All three consume a *sorted-ascending* stream of [`ApproxCandidate`]s
//! from the approximate stage and produce an exact-verified top-K.  The
//! only difference is how many candidates each policy is willing to pay
//! full-precision distance evaluations for.

use crate::pq::{ApproxCandidate, BruteForce, RerankResult};
use crate::DgarError;

/// Common interface implemented by every rerank policy.
///
/// Implementors return both the final top-K *and* the number of exact-distance
/// evaluations they consumed — the benchmark harness uses the latter as the
/// primary cost metric (it dominates PQ-scan cost on any real workload).
pub trait Reranker {
    /// Human-readable policy name — used for report grouping.
    fn name(&self) -> &'static str;

    /// Rerank `candidates` (must be sorted ascending by `distance`) and return
    /// the top-`k` by exact distance, along with the count of exact-distance
    /// evaluations performed.
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[ApproxCandidate],
        k: usize,
        exact: &BruteForce<'_>,
    ) -> Result<(Vec<RerankResult>, usize), DgarError>;
}

/// Classical fixed-multiplier reranker: evaluate the first `c * k` candidates.
pub struct FixedK {
    /// Multiplier applied to `k` at query time.
    pub c: usize,
}

impl Reranker for FixedK {
    fn name(&self) -> &'static str {
        "FixedK"
    }
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[ApproxCandidate],
        k: usize,
        exact: &BruteForce<'_>,
    ) -> Result<(Vec<RerankResult>, usize), DgarError> {
        if k > candidates.len() {
            return Err(DgarError::KTooLarge {
                k,
                n: candidates.len(),
            });
        }
        let target = (self.c * k).min(candidates.len());
        rerank_head(query, &candidates[..target], k, exact)
    }
}

/// Distance-Gap Adaptive Rerank.
///
/// Walk the candidate stream in ascending approximate distance.  Reject
/// (i.e. stop paying for exact distances) at the first index `i > k` where
///
/// `candidates[i].distance > (1 + gamma) * candidates[k-1].distance`
///
/// while keeping `i <= max_k * k` as a hard ceiling to bound worst-case
/// cost.  `min_k` (usually `= k`) is a floor to prevent under-provisioning
/// in the tiny-K regime.
pub struct AdaptiveGap {
    /// Multiplicative gap threshold. Typical range 0.02..0.30.
    pub gamma: f32,
    /// Hard ceiling on candidates to rerank, as a multiple of `k`.
    pub max_k: usize,
    /// Lower bound on candidates to rerank, as a multiple of `k`.
    pub min_k: usize,
}

impl AdaptiveGap {
    /// Update `gamma` at runtime (from e.g. a recall control loop).
    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }
}

impl Reranker for AdaptiveGap {
    fn name(&self) -> &'static str {
        "AdaptiveGap"
    }
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[ApproxCandidate],
        k: usize,
        exact: &BruteForce<'_>,
    ) -> Result<(Vec<RerankResult>, usize), DgarError> {
        if k > candidates.len() {
            return Err(DgarError::KTooLarge {
                k,
                n: candidates.len(),
            });
        }
        let anchor = candidates[k - 1].distance.max(1e-12);
        let ceiling = (self.max_k * k).min(candidates.len());
        let floor = (self.min_k * k).min(candidates.len()).max(k);
        let threshold = (1.0 + self.gamma) * anchor;
        let mut take = floor;
        for i in floor..ceiling {
            if candidates[i].distance > threshold {
                break;
            }
            take = i + 1;
        }
        rerank_head(query, &candidates[..take], k, exact)
    }
}

/// Oracle policy that only pays for exact distances on the *true* top-K.
///
/// This is not deployable — it presupposes ground truth — but it lets us
/// bound the information-theoretic minimum exact-eval cost.  Any deployable
/// reranker with strictly fewer evals is only doing so by dropping recall.
pub struct OracleUpperBound<'a> {
    /// Ground-truth top-K ids for the query.
    pub truth_ids: &'a [u32],
}

impl<'a> Reranker for OracleUpperBound<'a> {
    fn name(&self) -> &'static str {
        "OracleUpperBound"
    }
    fn rerank(
        &self,
        query: &[f32],
        _candidates: &[ApproxCandidate],
        k: usize,
        exact: &BruteForce<'_>,
    ) -> Result<(Vec<RerankResult>, usize), DgarError> {
        let take = self.truth_ids.len().min(k);
        let mut out: Vec<RerankResult> = self.truth_ids[..take]
            .iter()
            .map(|&id| RerankResult {
                id,
                distance: exact.distance(query, id),
            })
            .collect();
        out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok((out, take))
    }
}

fn rerank_head(
    query: &[f32],
    head: &[ApproxCandidate],
    k: usize,
    exact: &BruteForce<'_>,
) -> Result<(Vec<RerankResult>, usize), DgarError> {
    let mut scored: Vec<RerankResult> = head
        .iter()
        .map(|c| RerankResult {
            id: c.id,
            distance: exact.distance(query, c.id),
        })
        .collect();
    scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    scored.truncate(k);
    Ok((scored, head.len()))
}
