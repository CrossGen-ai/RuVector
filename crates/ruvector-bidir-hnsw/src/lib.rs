//! Bidirectional HNSW: Reverse-Neighbor Densification for HNSW graph search.
//!
//! Standard HNSW stores an out-adjacency list per node. When node `u` is inserted
//! and picks `v` as a neighbor, only `u -> v` is recorded (unless `v`'s list is
//! not full, in which case a symmetric edge `v -> u` is added). Reverse links —
//! "who pointed to me?" — are discarded during search.
//!
//! This crate implements a **reverse-neighbor sidecar**: after HNSW build, we
//! materialize the top-`R` in-degree neighbors per node into a compact `RevAdj`
//! table, then expose three search modes:
//!
//! 1. `Mode::Baseline`       — vanilla HNSW beam search (forward edges only).
//! 2. `Mode::ExpandReverse`  — at query time, when we visit node `u`, expand
//!                             candidates by the union of `Adj[u] ∪ RevAdj[u]`.
//! 3. `Mode::SymmetrizeBuild`— pre-symmetrize the graph so `Adj[u]` already
//!                             contains reverse edges (bounded to `M_max`).
//!
//! We measure recall@10 and cost (distance computations) for all three modes on
//! a synthetic Gaussian dataset with a held-out exact ground truth.
//!
//! No mocks. All numbers reported by the benchmark binary are real
//! `cargo run --release` measurements.

pub mod graph;
pub mod build;
pub mod search;
pub mod bench;

pub use graph::{BidirGraph, RevAdj};
pub use search::{search, Mode, SearchStats};
pub use build::{build_hnsw_like, BuildParams};
