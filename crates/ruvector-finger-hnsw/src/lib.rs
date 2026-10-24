//! ruvector-finger-hnsw
//!
//! FINGER-style low-rank edge sketches for cheap distance approximation on
//! HNSW-like graphs. For each directed edge (c -> n_i) we precompute a
//! Johnson-Lindenstrauss projection of the edge vector (n_i - c) onto a
//! shared random basis of rank k, plus the squared edge length. At query
//! time we compute the projection of (q - c) once per visited node and
//! approximate the inner product <q - c, n_i - c> from the two sketches,
//! giving an estimated distance d(q, n_i) in O(k) rather than O(d).
//!
//! The crate ships three swappable search backends behind the `Search`
//! trait so a caller can measure recall/latency tradeoffs:
//!
//!   - `ExactSearch`   — standard best-first graph search, full L2 at
//!                       every candidate.
//!   - `FingerSearch`  — JL-sketch approximation only (fast, lossy).
//!   - `FingerGuarded` — JL approximation as a lower-bound guard;
//!                       exact L2 is computed only when the approximation
//!                       could beat the current kth-best threshold.
//!
//! The graph itself is a flat kNN graph (brute-force neighbors per node)
//! so recall is identical across backends when they fall through to
//! exact distance. The point of this PoC is to measure the *distance
//! computation savings* of FINGER-style sketches, not the HNSW layering,
//! which is orthogonal.

pub mod graph;
pub mod sketch;
pub mod search;
pub mod data;

pub use graph::{FlatGraph, GraphParams};
pub use sketch::{EdgeSketch, SketchParams};
pub use search::{Search, SearchResult, SearchStats, ExactSearch, FingerSearch, FingerGuarded};
pub use data::{gen_dataset, gen_queries_from_data, brute_force_topk, recall_at_k};

/// Simple error type.
#[derive(Debug)]
pub enum Error {
    DimMismatch { expected: usize, got: usize },
    EmptyDataset,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimMismatch { expected, got } => {
                write!(f, "dim mismatch: expected {}, got {}", expected, got)
            }
            Error::EmptyDataset => write!(f, "empty dataset"),
        }
    }
}

impl std::error::Error for Error {}
