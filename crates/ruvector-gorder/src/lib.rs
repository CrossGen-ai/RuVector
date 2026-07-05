//! ruvector-gorder — Cache-Aware Node Reordering for HNSW-style Proximity Graphs
//!
//! This crate implements three swappable **node-ID layout** strategies for
//! HNSW-style graphs and measures their effect on search-time cache locality.
//!
//! The core idea (SOTA background): once an HNSW graph is built, the *order*
//! in which nodes are laid out in memory is independent from graph
//! correctness but dominates L1/L2/LLC miss rates during greedy graph
//! traversal. Reordering IDs so that neighbors are stored close in memory
//! is a well-known cache trick from graph analytics (Wei et al., "Speedup
//! Graph Processing by Graph Ordering", SIGMOD 2016 — a.k.a. **Gorder**).
//! Applying this to ANN graphs has recently been explored by NVIDIA CAGRA
//! and by cache-aware variants of DiskANN; ruvector did not have it.
//!
//! We implement three layouts behind a single trait:
//! * `InsertionLayout`     — baseline: id == insertion order
//! * `BfsLayout`           — breadth-first from the entry point
//! * `GorderLayout`        — sliding-window greedy Gorder
//!
//! And expose a small `MiniHnsw` graph so we can benchmark all three end-to-end
//! with a real greedy `search()` on the same graph structure.

pub mod graph;
pub mod layout;
pub mod search;

pub use graph::{MiniHnsw, MiniHnswParams};
pub use layout::{
    apply_permutation, BfsLayout, GorderLayout, InsertionLayout, Layout, LayoutStats,
};
pub use search::{search_greedy, SearchStats};
