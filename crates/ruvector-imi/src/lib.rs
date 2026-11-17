//! ruvector-imi: Inverted Multi-Index (IMI) coarse quantizer.
//!
//! Reference: Babenko & Lempitsky, "The Inverted Multi-Index", CVPR 2012.
//!
//! An IMI splits a d-dim vector into two halves and runs k-means on each
//! independently, producing two codebooks `U = {u_1..u_K}` and
//! `V = {v_1..v_K}`. The effective cell count is K^2 — a 1024-centroid
//! two-way IMI gives 1,048,576 virtual cells with only 2*1024 physical
//! centroids, so each cell averages N / K^2 points instead of the flat
//! IVF's N / K. Smaller cells + a multi-sequence enumerator that visits
//! cells in increasing combined distance give much smaller scanned
//! candidate sets at matched recall.

pub mod kmeans;
pub mod imi;
pub mod flat_ivf;
pub mod data;

pub use imi::{ImiIndex, ImiHit};
pub use flat_ivf::FlatIvf;
pub use kmeans::{kmeans, l2_sq};

/// Index trait: all variants expose the same ANN API so the benchmark
/// can swap them in and compare on equal footing.
pub trait AnnIndex {
    fn search(&self, query: &[f32], probe: usize, k: usize) -> Vec<(u32, f32)>;
    fn name(&self) -> &str;
    /// Bytes of state held by the index (centroids + postings).
    fn memory_bytes(&self) -> usize;
}
