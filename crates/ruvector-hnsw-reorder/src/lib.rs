//! ruvector-hnsw-reorder
//!
//! Cache-locality node reordering for HNSW-style ANN graphs.
//!
//! Given a built graph + vector store, we relabel node IDs so that graph
//! neighbors and search-time visit sequences touch nearby memory. The
//! search algorithm is unchanged; only the physical layout differs.
//!
//! Strategies:
//! - `Identity`   : original order (baseline)
//! - `Bfs`        : breadth-first from a random seed
//! - `Gorder`     : sliding-window co-access frequency (Wei & Karypis 2016)
//! - `Rgb`        : recursive graph bisection minimising log-gap cost
//!                  (Dhulipala et al. 2016 / Chierichetti et al. 2009)
//!
//! All strategies are pure-Rust, deterministic given a seed, and operate
//! on the CSR-style adjacency built by `HnswGraph::build`.

pub mod build;
pub mod graph;
pub mod reorder;
pub mod search;

pub use build::build_hnsw;
pub use graph::{HnswGraph, Vector};
pub use reorder::{apply_permutation, bfs_order, gorder, identity_order, rgb_order, Strategy};
pub use search::{search_knn, SearchStats};

#[cfg(test)]
mod tests;
