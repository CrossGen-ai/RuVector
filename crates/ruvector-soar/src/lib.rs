//! # ruvector-soar
//!
//! **SOAR** — *Spilling with Orthogonality-Amplified Residuals* — is an
//! anisotropic-loss duplicate-assignment strategy for IVF partition indexes,
//! introduced by Sun, Simcha, Dopson, Guo, Kumar & Xu at Google Research
//! (ICML 2024, "SOAR: Improved Indexing for Approximate Nearest Neighbor
//! Search"). SOAR generalizes SPANN's residual-ratio spill decision: instead
//! of always duplicating each point into its second-nearest centroid, SOAR
//! chooses the secondary centroid that minimizes an *anisotropic* loss that
//! penalizes duplicates whose displacement is parallel to the primary
//! residual — because a duplicate that "points the same direction" as the
//! primary partition provides no new query-time information.
//!
//! ## Variants implemented
//!
//! This crate provides three `PartitionIndex` variants under a common trait:
//!
//! - [`BaselineIvf`] — hard IVF, one partition per point (control).
//! - [`RandomSpillIvf`] — SPANN-style isotropic spill: each point in its top-2
//!   nearest partitions.
//! - [`SoarIvf`] — SOAR anisotropic spill: primary + orthogonality-selected
//!   secondary using the SOAR loss with regularization `lambda`.
//!
//! ## SOAR loss
//!
//! Given primary centroid `c1` (nearest to `x`) and candidate secondary `c2`,
//! let `r = x - c1` (primary residual) and `d = x - c2`. SOAR chooses `c2*`
//! minimizing:
//!
//! ```text
//! L(c2) = ||d||^2  +  (lambda - 1) * <d, r/||r||>^2
//! ```
//!
//! With `lambda = 1.0` this reduces to plain second-nearest (SPANN). With
//! `lambda > 1` (paper uses `lambda ≈ 3–4`), duplicates that lie *along*
//! the primary residual direction are penalized — because at query time,
//! queries near `x` will already probe `c1` for that direction, so a
//! collinear duplicate wastes memory. Orthogonal duplicates cover the
//! *other* query directions where `c1` alone is a poor proxy.
//!
//! ## Design
//!
//! Zero-dependency (`alloc`-only), deterministic, no `unsafe`. All three
//! variants share the same `PartitionIndex` trait so callers can swap them
//! by config. Vectors are `f32` slices in row-major layout.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod index;
pub mod kmeans;
pub mod rng;
pub mod vec_math;

pub use index::{
    BaselineIvf, PartitionIndex, PartitionStats, RandomSpillIvf, SearchResult, SoarIvf,
};
pub use kmeans::{kmeans_pp, KMeansConfig, KMeansModel};
pub use rng::Xorshift64;
pub use vec_math::{dot, l2_sq, sub_into};
