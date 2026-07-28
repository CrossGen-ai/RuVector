//! # ruvector-adsampling
//!
//! ADSampling — random-projection + adaptive early-termination distance
//! computation for approximate nearest-neighbor search. Based on
//! Gao & Long, "High-Dimensional Approximate Nearest Neighbor Search:
//! with Reliable and Efficient Distance Comparison Operations",
//! SIGMOD 2023 (paper 2306.11182).
//!
//! ## Idea in one paragraph
//!
//! Any ANN traversal (HNSW, IVF, brute-force top-k) spends the majority of
//! its wall time on `d`-dimensional L2 distance computations against a
//! rolling candidate set, but the *outcome* of nearly every comparison is
//! `> tau` — the candidate is worse than the current k-th best. ADSampling
//! (a) applies a single random orthonormal rotation R to both queries and
//! database vectors so that the accumulated partial-sum along the first
//! `m` dimensions is an unbiased estimator of the full squared distance,
//! and (b) walks that partial sum in blocks of size `delta`, aborting the
//! computation as soon as the *lower confidence bound* on the full
//! distance already exceeds the current top-k threshold `tau`. Vectors
//! that survive the entire walk fall back to the exact squared distance.
//! On random rotations the estimator is dimension-free — the sampling
//! bound depends only on `m/d` and `epsilon`, not on the vector norm —
//! so the pruning stays sound as `d` grows.
//!
//! ## What this crate provides
//!
//! * `RandomRotation`     — deterministic orthonormal rotation via a fixed
//!                          seed and Householder reflections (no BLAS
//!                          dependency).
//! * `AdsIndex`           — a rotated brute-force top-k index that plugs
//!                          into any downstream ANN as the *distance
//!                          function* rather than as a graph.
//! * `DistanceOracle`     — trait for a swappable comparator. Three
//!                          implementations ship: `ExactL2`,
//!                          `AdsFixedBudget` (naive fixed-`m` sample),
//!                          `AdsAdaptive` (the real algorithm, Alg. 1 of
//!                          the paper).
//! * `bench_variants`     — a real, non-mocked harness that produces
//!                          throughput / recall / distance-op counts for
//!                          the three variants against a shared corpus.
//!
//! ## Non-goals
//!
//! * No HNSW graph is built here — this crate is deliberately the
//!   *pruning layer*. ADR-273 explains how to wire `DistanceOracle` into
//!   the existing `ruvector-coherence-hnsw` traversal without changing
//!   its graph structure.
//! * No SIMD-specific intrinsics. The point is to show that the algorithm
//!   *asymptotically* removes work; SIMD would improve baseline exact-L2
//!   more than ADS and would confuse the recall/pruning story.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod oracle;
pub mod rotation;
pub mod index;
pub mod bench_variants;

pub use oracle::{DistanceOracle, ExactL2, AdsFixedBudget, AdsAdaptive, OracleStats};
pub use rotation::RandomRotation;
pub use index::{AdsIndex, TopK, Neighbor};
pub use bench_variants::{VariantReport, bench_variants};

/// Errors surfaced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum AdsError {
    /// Query and index dimensions disagree.
    #[error("dimension mismatch: index d={index_d}, query d={query_d}")]
    DimensionMismatch {
        /// Index dimensionality.
        index_d: usize,
        /// Query dimensionality.
        query_d: usize,
    },
    /// Requested top-k is zero.
    #[error("top-k must be >= 1")]
    ZeroK,
    /// Requested step size `delta` is zero or larger than `d`.
    #[error("invalid delta: got {delta}, must be in 1..=d ({d})")]
    InvalidDelta {
        /// Requested step size.
        delta: usize,
        /// Vector dimensionality.
        d: usize,
    },
}
