//! # ruvector-adaptive-ef
//!
//! Per-query adaptive `ef` selection for HNSW-style approximate nearest
//! neighbour search.  The classical HNSW search algorithm exposes one
//! knob — `ef_search` (a.k.a. `ef`) — which controls the size of the
//! candidate priority queue during graph traversal.  Larger `ef` ⇒ higher
//! recall but higher latency.
//!
//! In practice, query difficulty varies dramatically: out-of-distribution
//! (OOD) queries, queries near cluster boundaries, and high-density region
//! queries need larger `ef` to hit the same recall target than easy
//! in-distribution queries do.  A *fixed* `ef` either over-spends on easy
//! queries or under-spends on hard ones.
//!
//! This crate implements the SOTA idea of **per-query adaptive ef**:
//!
//! * [`EfPredictor`] — trait that maps a query vector + cheap online
//!   features (entry-point distances, neighbourhood density) to an `ef`.
//! * [`FixedEf`] — baseline.
//! * [`HeuristicAdaptiveEf`] — rule-based predictor inspired by LAET
//!   (Learning Adaptive Entry-points and Termination).
//! * [`LearnedAdaptiveEf`] — a tiny linear regressor (closed-form ridge)
//!   trained offline on `(features, min_ef_needed_to_hit_recall_target)`
//!   pairs — Auncel-style (SIGMOD 2023).
//!
//! The crate is self-contained: it ships its own minimal HNSW search
//! kernel ([`MiniHnsw`]) so it can be benchmarked without dragging the
//! whole `ruvector-core` workspace into the build.

pub mod features;
pub mod hnsw;
pub mod predictor;

pub use features::{QueryFeatures, extract_features};
pub use hnsw::{MiniHnsw, MiniHnswBuilder, SearchStats};
pub use predictor::{
    EfPredictor, FixedEf, HeuristicAdaptiveEf, LearnedAdaptiveEf, TrainingSample,
};
