//! ruvector-laet — Learned Adaptive Early Termination for HNSW.
//!
//! Given a tiny per-query feature vector, predict the smallest `ef`
//! that will still hit the recall target. Three search strategies are
//! exposed via the [`SearchStrategy`] trait so users can swap them:
//!
//! * [`FixedEfStrategy`] — classic baseline (uniform `ef`).
//! * [`GapHeuristicStrategy`] — dynamic early stop based on the
//!   distance gap between current best and the next candidate, no
//!   learning required.
//! * [`LaetStrategy`] — learned ridge-regression predictor that maps
//!   per-query features to an `ef` budget, calibrated offline.
//!
//! See `examples/bench.rs` for a runnable comparison.

pub mod data;
pub mod hnsw;
pub mod predictor;
pub mod search;

pub use data::{gen_clustered, l2sq, Dataset};
pub use hnsw::{Hnsw, HnswParams};
pub use predictor::{LaetFeatures, RidgePredictor};
pub use search::{
    FixedEfStrategy, GapHeuristicStrategy, LaetStrategy, SearchOutcome, SearchStrategy,
};
