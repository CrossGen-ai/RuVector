//! ruvector-adaptive-beam — Anytime ANN search via online quantile-based
//! early termination in HNSW.
//!
//! # Motivation
//!
//! Classical HNSW expands a beam of size `ef_search` until the beam top
//! candidate is worse than the worst element of the current top-k. With
//! a fixed `ef_search`, latency is roughly constant per query, but recall
//! gain per additional expansion has sharply diminishing returns. Many
//! queries reach near-final recall at ~30% of the budget.
//!
//! This crate ships a self-contained HNSW implementation with a pluggable
//! `BeamTerminator` trait. Three terminators are provided:
//!
//! * `FixedEfTerminator`  — classical HNSW.
//! * `RatioTerminator`    — early-stop when `min_unexpanded / worst_topk`
//!                          exceeds a fixed multiplier.
//! * `QuantileTerminator` — online P² quantile estimator over recent
//!                          "expansion improvement deltas"; stop when the
//!                          probability that the next expansion improves
//!                          the top-k drops below a threshold.
//!
//! All variants honour an `ef_max` ceiling so worst-case latency is bounded.
//!
//! See `docs/research/nightly/2026-06-21-adaptive-beam-hnsw/README.md`.

pub mod hnsw;
pub mod p2;
pub mod terminator;

pub use hnsw::{Hnsw, HnswParams, Neighbor};
pub use p2::P2Quantile;
pub use terminator::{
    BeamTerminator, FixedEfTerminator, QuantileTerminator, RatioTerminator, TerminationStats,
};
