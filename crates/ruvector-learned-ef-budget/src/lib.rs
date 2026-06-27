//! ruvector-learned-ef-budget
//!
//! Per-query learned beam-width predictor for HNSW. The classic HNSW search
//! algorithm uses a single static `ef_search` value chosen by the operator —
//! large enough to satisfy the worst-case query at the target recall, which
//! wastes work on easy queries. This crate ships a tiny linear regressor that
//! reads cheap per-query features (norm, distance to a small set of medoids,
//! and the entry-point's neighbourhood spread) and predicts the minimum `ef`
//! that will reach a target recall — the *Auncel/SLIM* family of techniques,
//! plus a "Steiner-hardness"-style hardness signal (LIMIT NeurIPS 2024).
//!
//! All distance computations are counted, so benchmarks report true work, not
//! wall-clock noise. The crate is self-contained: a minimal but correct HNSW
//! implementation lives in [`hnsw`] so we never have to coordinate the
//! adaptive logic with an upstream graph index.
//!
//! Modules:
//! - [`hnsw`]: minimal correct HNSW with explicit ef + distance counter
//! - [`features`]: per-query feature extraction
//! - [`predictor`]: closed-form ridge-regression budget predictor

pub mod features;
pub mod hnsw;
pub mod predictor;

pub use features::{extract_features, QueryFeatures, FEATURE_DIM};
pub use hnsw::{Hnsw, HnswParams, SearchStats};
pub use predictor::{BudgetPredictor, OracleBudget, PredictorConfig};

/// Library-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("empty index")]
    Empty,
    #[error("training data is empty")]
    NoTraining,
}

