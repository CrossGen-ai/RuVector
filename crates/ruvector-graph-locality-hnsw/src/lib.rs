//! Graph-locality reordering for HNSW (nightly research 2026-08-30, ADR-341).
//!
//! A minimal, self-contained HNSW index whose vector store is a flat `Vec<f32>`
//! with a physical-id -> logical-id permutation. Reordering strategies rearrange
//! the physical layout to improve cache locality during graph traversal.
//!
//! Strategies live behind the [`ReorderStrategy`] trait. Three implementations
//! are provided: [`IdentityOrder`] (baseline), [`BfsOrder`], [`RcmOrder`]
//! (Reverse Cuthill-McKee over the HNSW L0 adjacency).

#![forbid(unsafe_code)]

pub mod hnsw;
pub mod order;
pub mod query;

pub use hnsw::{HnswConfig, HnswIndex};
pub use order::{BfsOrder, IdentityOrder, RcmOrder, ReorderStrategy, Reordered};
pub use query::{search, SearchStats};

/// Compute squared Euclidean distance between two equal-length slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_basic() {
        let a = [0.0, 1.0, 2.0];
        let b = [1.0, 0.0, 3.0];
        assert!((sq_l2(&a, &b) - 3.0).abs() < 1e-6);
    }
}
