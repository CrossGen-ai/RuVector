//! DGAR: Distance-Gap Adaptive Rerank
//!
//! Two-stage ANN pipelines (PQ / RaBitQ / Matryoshka coarse → full-precision rerank)
//! typically fetch a fixed multiplier `K' = C * K` candidates from the approximate
//! stage, then compute exact distances for all of them.  The constant `C` is a
//! query-agnostic hyperparameter — over-provisioned for easy queries, under-
//! provisioned for hard ones.
//!
//! DGAR replaces the fixed multiplier with a query-adaptive rule based on the
//! *distance gap* between successively ranked approximate candidates.  The
//! intuition: once the ratio `d_approx[i] / d_approx[K-1]` exceeds a threshold
//! `1 + γ`, the probability that candidate `i` beats the current top-K after
//! reranking drops below a target level `α` (empirically calibrated).
//!
//! This crate ships three swappable `Reranker` implementations so pipelines can
//! swap policies at runtime:
//!
//! * [`FixedK`] — the classical baseline, `K' = C * K`.
//! * [`AdaptiveGap`] — DGAR, γ-thresholded truncation.
//! * [`OracleUpperBound`] — cheats by seeing the true top-K first; used as an
//!   information-theoretic ceiling to bound how much recall is left on the table.
//!
//! All three implement [`Reranker`] and share a single [`ApproxCandidate`] input
//! contract, so a serving layer can A/B them without touching the index.
//!
//! ## Failure modes
//!
//! * When the approximate distances are extremely noisy (very small codebook,
//!   very high dimensionality) the gap signal degenerates and AdaptiveGap
//!   collapses to `min_k = K`.  In that regime FixedK with a large `C` wins.
//! * When queries are drawn from an out-of-distribution shift the calibrated
//!   γ drifts.  AdaptiveGap exposes a `set_gamma` hook so a control loop can
//!   re-tune from live recall telemetry.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod policies;
mod pq;

pub use policies::{AdaptiveGap, FixedK, OracleUpperBound, Reranker};
pub use pq::{ApproxCandidate, BruteForce, ProductQuantizer, Query, RerankResult, Vector};

use thiserror::Error;

/// Errors returned by DGAR components.
#[derive(Debug, Error)]
pub enum DgarError {
    /// Requested `k` is larger than the number of candidates supplied by the
    /// approximate stage.
    #[error("k ({k}) exceeds candidate count ({n})")]
    KTooLarge {
        /// The requested top-`k`.
        k: usize,
        /// The number of candidates the approximate stage produced.
        n: usize,
    },
    /// Configuration was rejected before search began.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
