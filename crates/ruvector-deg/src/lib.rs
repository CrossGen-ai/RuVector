//! Dynamic Exploration Graph (DEG) — bounded-degree proximity graph supporting
//! streaming insert and delete with continuous edge refinement.
//!
//! Reference: Hezel, Schall, Jung & Barthel, "Fast Approximate Nearest Neighbor
//! Search with a Dynamic Exploration Graph using Continuous Refinement",
//! arXiv:2307.10479 (2023). This crate is an independent, simplified Rust
//! reimplementation focused on the core invariants:
//!
//!   * Every active vertex has exactly `degree` outgoing edges (bounded degree).
//!   * Edges are scored by edge weight (distance). Insertion and deletion both
//!     trigger local "edge optimisation" passes that try to replace heavy
//!     edges with lighter ones discovered during search.
//!   * Deletion is in-place: the slot is marked vacant, and any vertex that
//!     pointed at the deleted node patches its edge using the search results
//!     produced while disconnecting (no tombstone bloat).
//!
//! No external proximity-graph crate is used; this is a self-contained
//! implementation suitable as a building block for streaming agent memory.

pub mod distance;
pub mod graph;

pub use distance::Metric;
pub use graph::{DegGraph, DegParams, SearchStats};
