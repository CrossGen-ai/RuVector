//! # ruvector-specann — Speculative ANN Search
//!
//! Draft-and-verify vector retrieval inspired by LLM speculative decoding.
//!
//! **Idea.** A *draft index* (cheap, approximate — e.g. int8 or 1-bit signed)
//! proposes an over-provisioned candidate set of size `k_draft = alpha * k`.
//! A *verifier* (exact float32) rescores only that candidate set. An
//! *escalation policy* watches the confidence gap between the draft's top-k
//! and next candidates; when the gap is thin, the driver widens the draft
//! probe (larger `k_draft`) or falls back to exhaustive verification.
//!
//! This mirrors LLM speculative decoding: a small model drafts N tokens; a
//! big model verifies in one forward pass; accept until the first mismatch.
//! Here, draft and verifier operate over vectors, not tokens.
//!
//! ## Traits
//! - [`DraftIndex`]  — proposes candidates with a cheap score.
//! - [`Verifier`]    — exact rescoring of a candidate set.
//! - [`SpecAnnIndex`] — driver combining draft + verifier + escalation.
//!
//! ## Provided implementations (in [`backends`])
//! - [`F32BruteForce`]  — exact float32 brute force (baseline draft OR verifier).
//! - [`Int8BruteForce`] — symmetric int8 quantized brute force (fast draft).
//! - [`Sign1BitDraft`]  — 1-bit sign approximation (ultra-fast draft, RaBitQ-lite).

use std::cmp::Ordering;

use thiserror::Error;

pub mod backends;
pub use backends::{F32BruteForce, Int8BruteForce, Sign1BitDraft};

#[derive(Debug, Error)]
pub enum SpecAnnError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("empty index")]
    Empty,
}

pub type Result<T> = std::result::Result<T, SpecAnnError>;

/// A `(id, score)` pair. Lower `score` is closer for Euclidean-style metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub score: f32,
}

impl Eq for Neighbor {}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A cheap approximate index proposing over-provisioned candidate sets.
pub trait DraftIndex: Send + Sync {
    /// Return the top `k_draft` candidates for `query`, sorted by ascending draft score.
    fn draft(&self, query: &[f32], k_draft: usize) -> Result<Vec<Neighbor>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An exact rescorer over a candidate set.
pub trait Verifier: Send + Sync {
    /// Rescore `candidates` exactly against `query`.
    fn verify(&self, query: &[f32], candidates: &[u32]) -> Result<Vec<Neighbor>>;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
}

/// Escalation policy: decides whether a draft's top-k is trustworthy or must widen.
#[derive(Debug, Clone, Copy)]
pub struct EscalationPolicy {
    /// Over-provisioning multiplier: k_draft = alpha * k
    pub alpha: usize,
    /// Relative gap threshold. If (score[k] - score[k-1]) / score[k-1] < gap, escalate.
    pub gap_threshold: f32,
    /// Max escalations before falling back to full-index verify.
    pub max_escalations: u8,
    /// Multiplier applied to alpha on each escalation.
    pub escalate_multiplier: f32,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            alpha: 4,
            gap_threshold: 0.02,
            max_escalations: 2,
            escalate_multiplier: 2.5,
        }
    }
}

/// Aggregate stats returned by a speculative search — useful for research/telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    pub escalations: u8,
    pub draft_candidates: usize,
    pub verified: usize,
    pub full_verify_fallback: bool,
}

/// Driver: draft + verifier + escalation.
pub struct SpecAnnIndex<D: DraftIndex, V: Verifier> {
    pub draft: D,
    pub verifier: V,
    pub policy: EscalationPolicy,
}

impl<D: DraftIndex, V: Verifier> SpecAnnIndex<D, V> {
    pub fn new(draft: D, verifier: V, policy: EscalationPolicy) -> Self {
        Self {
            draft,
            verifier,
            policy,
        }
    }

    /// Speculative top-k search. Returns exact top-k neighbors + stats.
    pub fn search(&self, query: &[f32], k: usize) -> Result<(Vec<Neighbor>, SpecStats)> {
        if self.verifier.len() == 0 {
            return Err(SpecAnnError::Empty);
        }
        if query.len() != self.verifier.dim() {
            return Err(SpecAnnError::DimMismatch {
                expected: self.verifier.dim(),
                got: query.len(),
            });
        }

        let mut alpha_f = self.policy.alpha as f32;
        let mut stats = SpecStats::default();

        for step in 0..=self.policy.max_escalations {
            let k_draft = ((alpha_f as usize).max(1) * k).min(self.draft.len());
            let draft = self.draft.draft(query, k_draft)?;
            stats.draft_candidates = draft.len();

            // Confidence signal: relative gap around the k-th draft score.
            let confident = if draft.len() > k {
                let a = draft[k - 1].score.max(1e-12);
                let b = draft[k].score;
                ((b - a) / a) >= self.policy.gap_threshold
            } else {
                true
            };

            if confident || step == self.policy.max_escalations {
                let ids: Vec<u32> = draft.iter().map(|n| n.id).collect();
                let mut verified = self.verifier.verify(query, &ids)?;
                verified.sort();
                verified.truncate(k);
                stats.verified = ids.len();
                stats.escalations = step;
                return Ok((verified, stats));
            }
            alpha_f *= self.policy.escalate_multiplier;
        }

        // Fallback: verify entire index (rare).
        let all: Vec<u32> = (0..self.verifier.len() as u32).collect();
        let mut verified = self.verifier.verify(query, &all)?;
        verified.sort();
        verified.truncate(k);
        stats.full_verify_fallback = true;
        stats.verified = self.verifier.len();
        Ok((verified, stats))
    }
}

/// Convenience: recall@k of an approximate result against ground truth.
pub fn recall_at_k(approx: &[Neighbor], truth: &[Neighbor], k: usize) -> f32 {
    let k = k.min(truth.len()).min(approx.len());
    if k == 0 {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<u32> = truth.iter().take(k).map(|n| n.id).collect();
    let hit = approx.iter().take(k).filter(|n| truth_ids.contains(&n.id)).count();
    hit as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.sample(StandardNormal)).collect())
            .collect()
    }

    #[test]
    fn f32_bruteforce_matches_itself() {
        let vecs = synth(200, 32, 1);
        let bf = F32BruteForce::from_vectors(&vecs).unwrap();
        let q = &vecs[0];
        let top = bf.draft(q, 5).unwrap();
        assert_eq!(top[0].id, 0);
        assert!(top[0].score < 1e-5);
    }

    #[test]
    fn int8_draft_high_recall() {
        let vecs = synth(500, 64, 2);
        let bf = F32BruteForce::from_vectors(&vecs).unwrap();
        let q = synth(1, 64, 42).pop().unwrap();
        let truth = bf.draft(&q, 10).unwrap();
        let int8 = Int8BruteForce::from_vectors(&vecs).unwrap();
        let approx = int8.draft(&q, 10).unwrap();
        assert!(recall_at_k(&approx, &truth, 10) >= 0.7);
    }

    #[test]
    fn sign1bit_draft_reasonable_recall_when_over_provisioned() {
        let vecs = synth(500, 128, 3);
        let bf = F32BruteForce::from_vectors(&vecs).unwrap();
        let q = synth(1, 128, 99).pop().unwrap();
        let truth = bf.draft(&q, 10).unwrap();
        let sign = Sign1BitDraft::from_vectors(&vecs).unwrap();
        let approx = sign.draft(&q, 100).unwrap();
        let approx_ids: std::collections::HashSet<u32> = approx.iter().map(|n| n.id).collect();
        let hit = truth.iter().filter(|n| approx_ids.contains(&n.id)).count();
        assert!(hit >= 7, "expected ≥7/10 truth ids in 100-wide 1-bit draft, got {}", hit);
    }

    #[test]
    fn spec_ann_returns_exact_topk_matches_baseline() {
        let vecs = synth(1000, 64, 7);
        let baseline = F32BruteForce::from_vectors(&vecs).unwrap();
        let q = synth(1, 64, 11).pop().unwrap();
        let truth = baseline.draft(&q, 10).unwrap();

        let draft = Int8BruteForce::from_vectors(&vecs).unwrap();
        let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
        let spec = SpecAnnIndex::new(draft, verifier, EscalationPolicy::default());
        let (out, stats) = spec.search(&q, 10).unwrap();
        let r = recall_at_k(&out, &truth, 10);
        assert!(r >= 0.95, "spec-ann recall={}, stats={:?}", r, stats);
    }
}
