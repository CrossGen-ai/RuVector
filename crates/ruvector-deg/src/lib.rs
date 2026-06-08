//! ruvector-deg: Dynamic Exploration Graph
//!
//! A graph-based ANN index that natively supports insertions and deletions
//! without the long-running degradation observed in HNSW under churn.
//!
//! Design goals (per ADR-196):
//!   * Single-layer regular graph with edges per vertex = `d`
//!   * Edge-optimisation pass on insert that keeps the graph close to a
//!     RNG (Relative Neighbourhood Graph) approximation
//!   * O(1) tombstone-free delete via "swap-out" with the donor's neighbours
//!   * Pluggable distance via the `Metric` trait
//!
//! The implementation is intentionally < 500 lines and dependency-light so
//! that it can be embedded inside larger ruvector pipelines.

#![forbid(unsafe_code)]

pub mod distance;
pub mod graph;
pub mod search;

pub use distance::{L2, Metric};
pub use graph::{Deg, DegConfig, NodeId};
pub use search::SearchResult;
