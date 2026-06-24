//! Hubness-Aware HNSW (HUB-HNSW)
//!
//! A flat navigable-small-world graph index that addresses the *hubness*
//! phenomenon in high-dimensional ANN search: a small fraction of nodes
//! appear in disproportionately many k-NN lists and dominate graph traversal,
//! which both wastes work and skews recall.
//!
//! This crate provides three variants behind a single `AnnIndex` trait so
//! a downstream consumer can swap strategies and measure trade-offs:
//!
//! - [`BaselineNsw`] — standard NSW build, no hub mitigation.
//! - [`HubNsw`] with `IndegreeCap::Light` — soft cap at `3 * M` incoming edges.
//! - [`HubNsw`] with `IndegreeCap::Aggressive` — hard cap at `2 * M`.
//!
//! All variants share the same greedy beam-search inference path, so
//! reported deltas isolate the effect of anti-hub pruning.

pub mod graph;
pub mod hubness;
pub mod metric;

pub use graph::{AnnIndex, BaselineNsw, HubNsw, IndegreeCap, NswParams};
pub use hubness::{gini, indegree_histogram, IndegreeStats};
pub use metric::{l2_sq, Vector};
