//! Batched HNSW-style ANN with shared candidate frontier.
//!
//! Provides three search backends behind a common trait so they can be
//! swapped and benchmarked:
//!
//!   * [`independent::IndependentSearch`] — baseline; one greedy search per query.
//!   * [`batched::BatchedKernelSearch`]   — per-query heaps, but distance to each
//!     visited node is computed for **all** batch queries in one SIMD-friendly
//!     kernel, amortizing memory bandwidth on the candidate vector load.
//!   * [`shared::SharedFrontierSearch`]   — additionally dedupes candidate
//!     evaluations across the batch: if a node has already been evaluated for
//!     the batch, its cached per-query distance row is reused.
//!
//! All three operate on the same [`ProximityGraph`] and produce comparable
//! top-`k` results measured by [`BenchReport`].

pub mod dist;
pub mod graph;
pub mod independent;
pub mod batched;
pub mod shared;
pub mod bench;

pub use bench::BenchReport;
pub use graph::{ProximityGraph, ProximityGraphBuilder};

/// Result of a top-`k` ANN search for a single query.
#[derive(Debug, Clone)]
pub struct TopK {
    /// Indices into the graph's vector store, sorted by ascending distance.
    pub ids: Vec<u32>,
    /// Corresponding squared-L2 distances.
    pub dists: Vec<f32>,
}

impl TopK {
    pub fn empty() -> Self {
        Self { ids: Vec::new(), dists: Vec::new() }
    }
}

/// Trait implemented by every search backend. Kept small on purpose so a
/// production crate can swap backends per-workload.
pub trait BatchedSearch {
    /// Human-readable backend name (used in reports).
    fn name(&self) -> &'static str;

    /// Run top-`k` search for every query in `queries`. `queries.len()` is the
    /// batch size `Q`; each query is a `dim`-vector matching the graph.
    fn search_batch(
        &self,
        graph: &ProximityGraph,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<TopK>;
}
