//! ruvector-adaptive-ef: Query-adaptive `ef_search` for graph-based ANN.
//!
//! Most graph indexes (HNSW/NSW) expose an `ef_search` knob that linearly
//! controls the beam width during search. Setting it once per index forces a
//! single recall/latency tradeoff across *all* queries. Real workloads have
//! wildly different per-query difficulty:
//!
//! * a query landing in a dense, well-connected region finds its top-k with
//!   `ef = 32`
//! * a query near a cluster boundary or in a sparse region needs `ef = 256`
//!   for the same recall
//!
//! Using the max `ef` everywhere is the usual fix — and wastes 5–10× the
//! distance computations on easy queries.
//!
//! This crate provides:
//!
//! * [`NswIndex`] — a minimal single-layer Navigable Small World index. It is
//!   the part of HNSW where `ef_search` does the work; isolating it keeps the
//!   experiment clean and the crate under 500 lines.
//! * [`AdaptiveEf`] — a lightweight predictor trained from a small labelled
//!   batch (`query -> minimum ef to hit target recall`). At inference the
//!   predictor returns a per-query `ef`, clipped to `[ef_min, ef_max]`.
//!
//! The 3 measured variants in `main.rs` are:
//!   * `fixed_lo`  : a baseline at the smallest ef that hits target recall on
//!                   the *median* query (under-shoots tail recall).
//!   * `fixed_hi`  : the smallest ef that hits target recall on the *p95*
//!                   query (the textbook safe choice).
//!   * `adaptive`  : `AdaptiveEf` predicts per-query.
//!
//! Adaptive wins by matching `fixed_hi`'s recall at a fraction of its work.

pub mod adaptive;
pub mod nsw;

pub use adaptive::{AdaptiveEf, EfFeatures};
pub use nsw::{NswIndex, SearchStats};
