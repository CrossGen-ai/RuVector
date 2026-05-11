//! ruvector-soar: SOAR (Spillover-Optimized Anisotropic Residuals) for IVF.
//!
//! Reference: Sun, Simcha, Dopson, Guo, Kumar, "SOAR: Improved Indexing for
//! Approximate Nearest Neighbor Search", NeurIPS 2023.
//!
//! IVF assigns each database vector to its nearest centroid; recall depends on
//! probing the right cell. SOAR additionally assigns each vector to a SECOND
//! cell chosen so the secondary residual is orthogonally complementary to the
//! primary residual. The redundancy roughly doubles the effective recall per
//! nprobe at <2x storage overhead.

pub mod distance;
pub mod index;
pub mod kmeans;

pub use distance::{dot, l2_sq};
pub use index::{Assignment, IvfIndex, SearchResult, SoarConfig};
pub use kmeans::{kmeans_lloyd, KMeansConfig};

#[derive(thiserror::Error, Debug)]
pub enum SoarError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty dataset")]
    EmptyDataset,
    #[error("k ({k}) must be <= n ({n})")]
    KTooLarge { k: usize, n: usize },
}

pub type Result<T> = std::result::Result<T, SoarError>;
