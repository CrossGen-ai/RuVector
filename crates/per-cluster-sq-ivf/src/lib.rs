//! Per-Cluster 4-bit Scalar Quantization for IVF indexes.
//!
//! Three storage backends share a single `IvfBackend` trait so the IVF driver
//! is identical across them:
//!   * `Float32Backend`      - exact float32 residuals (baseline)
//!   * `GlobalSq4Backend`    - 4-bit SQ with one (min,max) per dimension
//!   * `PerClusterSq4Backend`- 4-bit SQ with one (min,max) per (cluster, dim)
//!
//! Everything is implemented against raw slices to keep the hot path tight.

pub mod kmeans;
pub mod sq;
pub mod ivf;
pub mod data;

pub use ivf::{IvfIndex, IvfBackend, Float32Backend, GlobalSq4Backend, PerClusterSq4Backend};
