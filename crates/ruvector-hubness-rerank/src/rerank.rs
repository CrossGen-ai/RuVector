//! Reranker implementations. Each rerank strategy takes a query and a
//! shortlist of `(id, raw_dist)` and returns a re-scored, re-sorted list.
//!
//! The convention is that lower scores are better for every strategy, so
//! we can compose them uniformly with `sort_by` at the end.

use crate::hubness::{empirical_survival, HubnessStats};

/// A single scored candidate.
#[derive(Copy, Clone, Debug)]
pub struct Scored {
    pub id: usize,
    pub score: f32,
}

/// Rerank interface. A reranker consumes the query and the coarse shortlist
/// and returns the top-`k` corrected scores. Implementations own whatever
/// precomputed state they need (usually a reference to `HubnessStats`).
pub trait Rerank {
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[Scored],
        query_dists_to_corpus: &[f32],
        k: usize,
    ) -> Vec<Scored>;

    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Baseline: raw distance, no correction.
// ---------------------------------------------------------------------------

pub struct BaselineRerank;

impl Rerank for BaselineRerank {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Scored],
        _query_dists_to_corpus: &[f32],
        k: usize,
    ) -> Vec<Scored> {
        let mut out = candidates.to_vec();
        out.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        out.truncate(k);
        out
    }
    fn name(&self) -> &'static str {
        "baseline"
    }
}

// ---------------------------------------------------------------------------
// Local Scaling (Zelnik-Manor & Perona 2004): d̂(q, v) = d(q, v) / sqrt(σ_q · σ_v)
// where σ_v is the k-th NN distance of v, and σ_q is the k-th NN distance
// of q in the coarse shortlist.
// ---------------------------------------------------------------------------

pub struct LocalScalingRerank<'a> {
    pub stats: &'a HubnessStats,
    /// k used to pick σ_q from the shortlist.
    pub k: usize,
}

impl<'a> Rerank for LocalScalingRerank<'a> {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Scored],
        _query_dists_to_corpus: &[f32],
        k: usize,
    ) -> Vec<Scored> {
        // σ_q = the k-th raw distance in the *shortlist*. We defensively
        // clamp k to the shortlist length; a caller with too-small R would
        // otherwise index OOB.
        let idx_k = self.k.min(candidates.len().saturating_sub(1));
        let mut sorted_c = candidates.to_vec();
        sorted_c.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        let sigma_q = sorted_c[idx_k].score.max(1e-9);

        let mut out: Vec<Scored> = candidates
            .iter()
            .map(|c| {
                let sigma_v = self.stats.kth_dist[c.id].max(1e-9);
                let denom = (sigma_q * sigma_v).sqrt();
                Scored {
                    id: c.id,
                    score: c.score / denom,
                }
            })
            .collect();
        out.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        out.truncate(k);
        out
    }
    fn name(&self) -> &'static str {
        "local_scaling"
    }
}

// ---------------------------------------------------------------------------
// NICDM (Jégou et al. 2010): d̂(q, v) = d(q, v) / sqrt(μ_q · μ_v)
// where μ_v is the arithmetic mean of the top-k neighbour distances of v.
// ---------------------------------------------------------------------------

pub struct NicdmRerank<'a> {
    pub stats: &'a HubnessStats,
    pub k: usize,
}

impl<'a> Rerank for NicdmRerank<'a> {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Scored],
        _query_dists_to_corpus: &[f32],
        k: usize,
    ) -> Vec<Scored> {
        // μ_q = arithmetic mean of the top-k shortlist distances.
        let idx_k = self.k.min(candidates.len());
        let mut sorted_c = candidates.to_vec();
        sorted_c.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        let mu_q: f32 =
            sorted_c[..idx_k].iter().map(|s| s.score).sum::<f32>() / idx_k.max(1) as f32;
        let mu_q = mu_q.max(1e-9);

        let mut out: Vec<Scored> = candidates
            .iter()
            .map(|c| {
                let mu_v = self.stats.topk_mean[c.id].max(1e-9);
                let denom = (mu_q * mu_v).sqrt();
                Scored {
                    id: c.id,
                    score: c.score / denom,
                }
            })
            .collect();
        out.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        out.truncate(k);
        out
    }
    fn name(&self) -> &'static str {
        "nicdm"
    }
}

// ---------------------------------------------------------------------------
// Mutual Proximity — Empiric (Schnitzer et al. 2012).
//
//   MP(q, v) = 1 - P(D_q > d(q,v)  ∧  D_v > d(v,q))
//
// Under an independence approximation over the two survival functions:
//
//   MP(q, v) ≈ 1 - S_q(d(q,v)) · S_v(d(q,v))
//
// Lower is better (dissimilarity). We form S_q from the query's coarse
// distances to the *corpus* (proxy for its neighbour distribution), and
// S_v from the precomputed sorted row for corpus[v].
// ---------------------------------------------------------------------------

pub struct MutualProximityRerank<'a> {
    pub stats: &'a HubnessStats,
}

impl<'a> Rerank for MutualProximityRerank<'a> {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Scored],
        query_dists_to_corpus: &[f32],
        k: usize,
    ) -> Vec<Scored> {
        // Sorted query distance distribution over the corpus.
        let mut q_sorted: Vec<f32> = query_dists_to_corpus.to_vec();
        q_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut out: Vec<Scored> = candidates
            .iter()
            .map(|c| {
                let d = c.score;
                let s_q = empirical_survival(&q_sorted, d);
                let s_v = empirical_survival(&self.stats.sorted_dists[c.id], d);
                let mp = 1.0 - s_q * s_v;
                Scored {
                    id: c.id,
                    score: mp,
                }
            })
            .collect();
        out.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        out.truncate(k);
        out
    }
    fn name(&self) -> &'static str {
        "mutual_proximity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hubness::hubness_stats;
    use crate::index::{FlatIndex, Vector};
    use crate::L2;

    fn small_corpus() -> Vec<Vector> {
        // A 1-D corpus with one obvious "hub" near the mean.
        vec![
            vec![-3.0],
            vec![-2.0],
            vec![-1.0],
            vec![0.0], // hub
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ]
    }

    #[test]
    fn baseline_returns_topk() {
        let c = small_corpus();
        let idx = FlatIndex::new(&c, L2);
        let q = vec![0.5];
        let coarse = idx.coarse_top_r(&q, 5);
        let out = BaselineRerank.rerank(&q, &coarse, &idx.distances_to(&q), 3);
        assert_eq!(out.len(), 3);
        // Under raw L2 the closest is index 3 (0.0).
        assert_eq!(out[0].id, 3);
    }

    #[test]
    fn all_rerankers_produce_k_results() {
        let c = small_corpus();
        let idx = FlatIndex::new(&c, L2);
        let stats = hubness_stats(&c, &L2, 2);
        let q = vec![0.5];
        let coarse = idx.coarse_top_r(&q, 6);
        let qd = idx.distances_to(&q);
        let rerankers: Vec<Box<dyn Rerank>> = vec![
            Box::new(BaselineRerank),
            Box::new(LocalScalingRerank {
                stats: &stats,
                k: 2,
            }),
            Box::new(NicdmRerank {
                stats: &stats,
                k: 2,
            }),
            Box::new(MutualProximityRerank { stats: &stats }),
        ];
        for r in &rerankers {
            let out = r.rerank(&q, &coarse, &qd, 3);
            assert_eq!(out.len(), 3, "reranker {} wrong k", r.name());
        }
    }
}
