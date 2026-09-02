//! Cache-oblivious HNSW graph layouts.
//!
//! Small HNSW implementation used to study how three memory layouts of the
//! same logical graph — BFS (baseline), DFS, and van Emde Boas (vEB) — affect
//! greedy-search latency and last-level cache traffic on modern CPUs.
//!
//! The graph structure is identical across variants; only the *node id → slot*
//! permutation differs. Vectors and adjacency lists are stored in flat arrays
//! indexed by the permuted slot, which is what the search hot path touches.
//!
//! This is a *research* crate: it exists to isolate layout effects, not to
//! replace `ruvector-hnsw`. It builds standalone (its own `[workspace]`).

pub mod build;
pub mod graph;
pub mod layout;
pub mod search;

pub use build::{build_hnsw, BuildParams};
pub use graph::{FlatGraph, Layout};
pub use layout::{bfs_permutation, dfs_permutation, veb_permutation};
pub use search::{greedy_search, SearchParams, SearchStats};

/// Vector dimension used throughout the research crate. Kept as a compile-time
/// constant so distance kernels can be inlined.
pub const DIM: usize = 128;
