//! ruvector-topk-stability-terminate — Kendall-tau top-k ordering stability
//! as an early-termination signal for approximate nearest-neighbor beam search.
//!
//! This crate is a self-contained research PoC (ADR-305). It provides:
//!
//!   * a small, deterministic k-NN-graph beam-search index (no HNSW dependency),
//!   * a pluggable [`TerminationPolicy`] trait,
//!   * three concrete policies: [`FixedBudget`] (baseline), [`GapThreshold`]
//!     (a common heuristic — terminate when the k-th distance stops improving),
//!     and [`KendallTauStability`] (this crate's contribution — terminate when
//!     the ordinal ranking of the current top-k has been stable for `w`
//!     consecutive iterations under Kendall's tau).
//!
//! The design is trait-based so the same query loop is used for every policy,
//! ensuring the benchmarks compare *policies* rather than incidental code
//! differences.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub mod graph;
pub mod policy;
pub mod search;
pub mod util;

pub use graph::KnnGraph;
pub use policy::{FixedBudget, GapThreshold, KendallTauStability, TerminationPolicy};
pub use search::{search, SearchStats};

/// A `(distance, node_id)` pair used inside the search's heaps.
///
/// Ordered by distance ascending under `Ord`, so we can put it in a
/// min-heap-of-candidates and a max-heap-of-results using `Reverse` / plain
/// respectively.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scored {
    pub dist: f32,
    pub id: u32,
}

impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Scored {
    fn cmp(&self, o: &Self) -> Ordering {
        // NaN-safe: fall back to id ordering if distances are non-comparable.
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&o.id))
    }
}

/// A convenience alias — used to make the search's candidate min-heap explicit.
pub type MinHeap = BinaryHeap<std::cmp::Reverse<Scored>>;
