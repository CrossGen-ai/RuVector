//! ruvector-early-term
//!
//! Trait-based HNSW search with three swappable termination policies:
//!   - Fixed `ef` (baseline)
//!   - Slope-based: stop when the k-th best distance has not improved
//!     by more than `eps` over a sliding window of `w` candidate expansions.
//!   - Learned linear predictor: extracts per-query features (mean/var of
//!     visited distances, gap to k-th best, progress ratio) and predicts
//!     remaining-recall risk. Trained offline on a held-out query set via
//!     ridge regression in closed form.
//!
//! The HNSW implementation is intentionally small and self-contained so the
//! research focus stays on the termination policy. It uses cosine distance
//! on L2-normalized vectors.

pub mod hnsw;
pub mod policy;
pub mod predictor;
pub mod data;

pub use hnsw::{Hnsw, HnswParams, SearchStats};
pub use policy::{TerminationPolicy, FixedEf, SlopePolicy, LearnedPolicy};
pub use predictor::{RidgeRegressor, QueryFeatures};
