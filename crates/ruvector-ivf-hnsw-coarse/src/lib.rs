//! ruvector-ivf-hnsw-coarse
//!
//! HNSW coarse quantizer for IVF. Compare three coarse quantizers behind the
//! [`IvfCoarse`] trait:
//!   * [`FlatCoarse`]      -- brute force, oracle recall
//!   * [`HnswCoarse`]      -- HNSW graph over centroids, O(log C) selection
//!   * [`RandomProjCoarse`] -- signed random projection bit-sketch, baseline
//!
//! An IVF index takes each coarse quantizer and delegates nprobe list
//! selection to it. Postings-list scan is identical across backends, so any
//! recall / latency difference is *purely* from the coarse quantizer.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod rproj;

pub use flat::FlatCoarse;
pub use hnsw::HnswCoarse;
pub use ivf::{Ivf, IvfConfig, IvfStats};
pub use rproj::RandomProjCoarse;

use thiserror::Error;

/// Errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum CoarseError {
    /// Configured dimension does not match the vectors provided.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Provided dimension.
        got: usize,
    },
    /// nprobe is larger than the centroid count.
    #[error("nprobe {nprobe} > centroids {c}")]
    NprobeTooLarge {
        /// Requested nprobe.
        nprobe: usize,
        /// Available centroids.
        c: usize,
    },
    /// Requested top-k is larger than the vector count.
    #[error("top-k {k} > n {n}")]
    KTooLarge {
        /// Requested k.
        k: usize,
        /// Available items.
        n: usize,
    },
}

/// Coarse quantizer contract. An IVF index calls [`IvfCoarse::top_lists`]
/// once per query to pick which posting lists (centroid IDs) to scan.
pub trait IvfCoarse: Send + Sync {
    /// Fit the quantizer over the given centroid rows.
    ///
    /// `centroids` is a row-major `C x d` slice. The trait implementation
    /// keeps whatever internal structure it needs to answer `top_lists`.
    fn fit(&mut self, centroids: &[f32], d: usize) -> Result<(), CoarseError>;

    /// Return the top-`nprobe` centroid IDs closest to `query` under
    /// squared-L2 distance, sorted ascending.
    fn top_lists(&self, query: &[f32], nprobe: usize) -> Vec<u32>;

    /// Human-readable backend name for reporting.
    fn name(&self) -> &'static str;

    /// Rough bytes of extra memory used *beyond* the centroid vectors.
    fn extra_bytes(&self) -> usize;
}

/// Squared-L2 helper used by every backend and by the postings scanner.
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
