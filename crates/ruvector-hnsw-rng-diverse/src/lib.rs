//! ruvector-hnsw-rng-diverse
//!
//! Comparative study of three neighbor-selection ("pruning") strategies for
//! graph-based ANN indices:
//!
//!   1. `Naive`      — keep the top-M nearest candidates.
//!   2. `Rng`        — Relative Neighborhood Graph pruning (Malkov, HNSW paper).
//!   3. `AlphaPrune` — Vamana / DiskANN α-pruning (Subramanya et al., 2019).
//!
//! Same base graph construction path; strategies differ only in the
//! `select_neighbors` step. This isolates the pruning contribution from every
//! other axis (entry point, graph degree, dataset).
//!
//! Public trait: [`Pruner`]. Public builders: [`build_index`], [`search`].
//!
//! No external crates, pure `std`. ~500 LoC total.

pub mod data;
pub mod graph;
pub mod prune;
pub mod search;

pub use graph::{build_index, GraphIndex};
pub use prune::{AlphaPrune, Naive, Pruner, RngPrune};
pub use search::{search, search_multi, SearchStats};

/// Squared euclidean distance (avoids a sqrt in the hot path).
#[inline]
pub fn dist2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests;
