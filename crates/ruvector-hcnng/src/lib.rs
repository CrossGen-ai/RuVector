//! HCNNG — Hierarchical Clustering-based Navigating Neighbor Graph.
//!
//! Reference: Munoz, Gonzalez, Buhmann (Pattern Recognition 2019/2022),
//! "Hierarchical Clustering-Based Graphs for Large Scale Approximate Nearest
//! Neighbor Search."
//!
//! Construction (per tree):
//!   1. Recursively partition the dataset by random 2-point pivot splits
//!      until every leaf has at most `leaf_size` points.
//!   2. On each leaf, build a Minimum Spanning Tree (Kruskal/Prim, O(L^2)).
//!   3. Union all MST edges into a global undirected proximity graph.
//! Repeat `n_trees` times. Final graph is the union of all per-tree edges,
//! truncated to `max_degree` nearest neighbors per node.
//!
//! Search uses a guided greedy walk (best-first with a small candidate heap)
//! that mirrors HNSW's level-0 search but on the HCNNG graph.

pub mod error;
pub mod distance;
pub mod partition;
pub mod mst;
pub mod graph;
pub mod search;
pub mod index;

pub use error::HcnngError;
pub use distance::{Distance, Metric};
pub use index::{HcnngIndex, HcnngParams};
pub use search::SearchResult;
