//! ruvector-learned-ef-predictor
//! =============================
//!
//! Query-adaptive `ef_search` controllers for HNSW-style ANN indexes.
//!
//! Three controllers implement a common [`EfController`] trait:
//! - [`FixedEf`]: constant `ef` baseline.
//! - [`GapRatioEf`]: heuristic on d2/d1 gap from a cheap probe.
//! - [`LearnedLinearEf`]: OLS-fit linear model on 6 probe features.
//!
//! See `examples/bench.rs` for a runnable comparison and `tests/` for
//! integration assertions.

pub mod calibrate;
pub mod controller;
pub mod dataset;
pub mod hnsw;

pub use calibrate::{calibrate, recall_at_k, EF_LADDER};
pub use controller::{EfController, FixedEf, GapRatioEf, LearnedLinearEf};
pub use dataset::{make_clusters, make_queries};
pub use hnsw::{Hnsw, ProbeStats};
