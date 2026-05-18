//! # ruvector-roargraph — Projected Bipartite Graph for OOD ANNS
//!
//! An implementation of RoarGraph (Chen et al., VLDB 2024), a query-aware graph
//! index that significantly improves approximate nearest-neighbour recall when
//! the **query distribution differs from the base distribution** — the common
//! case in cross-modal retrieval (text queries over image embeddings, CLIP, etc.).
//!
//! ## Core idea
//!
//! Classic graph-based ANN indices (HNSW, NSG, DiskANN) construct their graphs
//! using *base-to-base* distances.  When queries are out-of-distribution (OOD),
//! the greedy traversal enters the graph in the wrong region and recall degrades.
//!
//! RoarGraph instead:
//! 1. Samples a set of *training queries* from the query distribution.
//! 2. Builds a bipartite graph between training queries and base vectors (each
//!    query connects to its k-NN among base vectors).
//! 3. **Projects** the bipartite graph onto the base set: two base vectors
//!    become neighbours if they co-appear in any training query's neighbour list.
//! 4. Applies a BFS connectivity pass to ensure the graph is connected.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`error`] | `RoarError` enum |
//! | [`graph`] | `RoarGraph` struct + greedy beam search |
//! | [`build`] | Bipartite projection + connectivity pass |
//! | [`baseline`] | Base-to-base k-NN graph (OOD-naive baseline) |
//! | [`dataset`] | Synthetic OOD dataset generator + brute-force GT |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use ruvector_roargraph::{AnnIndex, RoarGraph, build::{BuildParams, build_roargraph}};
//!
//! let base: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//! let train_queries: Vec<Vec<f32>> = vec![vec![1.1, 2.1]];
//!
//! let mut idx = RoarGraph::new(2);
//! idx.add(&base).unwrap();
//! let params = BuildParams { k_train: 1, max_degree: 4 };
//! build_roargraph(&mut idx, &train_queries, &params).unwrap();
//! let results = idx.search(&[1.0, 2.0], 1, 10).unwrap();
//! assert_eq!(results[0].id, 0);
//! ```
//!
//! ## References
//!
//! - RoarGraph: Chen et al., *RoarGraph: A Projected Bipartite Graph for
//!   Efficient Cross-Modal Approximate Nearest Neighbor Search*, VLDB 2024.
//!   arXiv:2408.08933.
//! - HNSW: Malkov & Yashunin, 2018.
//! - DiskANN/Vamana: Subramanya et al., NeurIPS 2019.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod baseline;
pub mod build;
pub mod dataset;
pub mod error;
pub mod graph;

pub use baseline::BaselineGraph;
pub use build::{build_roargraph, BuildParams};
pub use error::RoarError;
pub use graph::{l2sq, RoarGraph, SearchResult};

/// Common trait implemented by both `RoarGraph` (via adaptor) and `BaselineGraph`.
///
/// `training_queries` is passed to `build()` but ignored by `BaselineGraph`
/// (which uses base-to-base distances only).
pub trait AnnIndex {
    /// Insert base vectors into the index.
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RoarError>;
    /// Construct the index.  For RoarGraph, `training_queries` drives the
    /// bipartite projection; for `BaselineGraph`, this argument is ignored.
    fn build(&mut self, training_queries: &[Vec<f32>]) -> Result<(), RoarError>;
    /// Greedy beam search returning up to `k` results with beam width `ef`.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, RoarError>;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
}

/// `AnnIndex` adaptor for the bare `RoarGraph` struct.
///
/// Delegates `build()` to [`build_roargraph`] with default [`BuildParams`].
pub struct RoarGraphIndex {
    pub(crate) inner: RoarGraph,
    pub(crate) params: BuildParams,
}

impl RoarGraphIndex {
    /// Create a new index with custom build parameters.
    pub fn new(dim: usize, params: BuildParams) -> Self {
        RoarGraphIndex {
            inner: RoarGraph::new(dim),
            params,
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl AnnIndex for RoarGraphIndex {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RoarError> {
        self.inner.add(vectors)
    }

    fn build(&mut self, training_queries: &[Vec<f32>]) -> Result<(), RoarError> {
        build_roargraph(&mut self.inner, training_queries, &self.params)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, RoarError> {
        self.inner.search(query, k, ef)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}
