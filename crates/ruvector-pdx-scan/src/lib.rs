//! # ruvector-pdx-scan
//!
//! PDX ("Partitioned Data eXchange" — dimension-partitioned vector storage)
//! layout for pruned brute-force and IVF top-k search.
//!
//! Reference: Kuffo, Krommydas, Boncz — *"PDX: A Data Layout for Vector
//! Similarity Search"*, SIGMOD 2024. This crate is a from-scratch Rust
//! implementation of the core idea plus a trait-based backend zoo so the
//! layout can be swapped underneath the same scan API.
//!
//! ## Idea in one paragraph
//!
//! Classical vector stores keep each vector's `d` dimensions contiguous
//! (row-major, "horizontal"). L2 / cosine kernels then load one vector at a
//! time and reduce across `d`. PDX flips the storage: it groups vectors into
//! *blocks* of `B` vectors (typically 64) and inside a block stores
//! dimension `j` for **all `B` vectors contiguously**. A block is a `d × B`
//! column-major tile.
//!
//! With this layout you can:
//!
//! 1. Load one SIMD-aligned `B`-lane vector per dimension and issue the same
//!    op for all `B` candidates simultaneously (no shuffles, no reductions
//!    inside the hot loop).
//! 2. Walk dimensions **in a pruning-friendly order** (e.g. by variance) and
//!    after every group of `dim_step` dimensions compare the running partial
//!    squared distance against the current top-k threshold; candidates whose
//!    partial already exceeds the threshold are dropped from the block's
//!    active mask before their remaining dimensions are ever touched.
//!
//! On modern CPUs this yields large brute-force / IVF-list speedups because
//! the scan is (a) bandwidth-bound on a *predictable* stride, and (b) does
//! most of its arithmetic on lanes that will never survive to the final
//! comparison.

pub mod block;
pub mod scan;
pub mod pruned;
pub mod ivf;

pub use block::{HorizontalStore, PdxStore, BLOCK_LEN};
pub use scan::{Backend, TopK, brute_horizontal, brute_pdx};
pub use pruned::{brute_pdx_pruned, brute_pdx_block_pruned, DimensionOrder, PruneStats};
pub use ivf::{IvfHorizontal, IvfPdxPruned, IvfStats};

/// Convenience wrapper: build a top-k heap-limited to `k` results.
pub fn topk(k: usize) -> TopK {
    TopK::new(k)
}
