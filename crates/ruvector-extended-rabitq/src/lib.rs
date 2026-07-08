#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]

//! Extended RaBitQ: Multi-bit Rotation-Based Scalar Quantization for ANN
//!
//! This crate implements a multi-bit generalization of the 1-bit RaBitQ
//! algorithm (Gao & Long, SIGMOD 2024) motivated by the follow-up paper
//! "Practical and Asymptotically Optimal Quantization of High-Dimensional
//! Vectors in Euclidean Space for Approximate Nearest Neighbor Search"
//! (Gao & Long, arXiv:2409.09913, 2024).
//!
//! # Idea in one paragraph
//!
//! 1-bit RaBitQ rotates every database vector by a Haar-uniform orthogonal
//! matrix `P`, unit-normalises the result, then keeps only the sign of each
//! coordinate. Under the rotation the sign vector becomes an unbiased +
//! low-variance estimator of the cosine between database vector and query.
//! Extended RaBitQ raises the alphabet size from 2 (±1) to `2^B` and uses a
//! symmetric uniform grid `{ (2k+1)/2^B − 1 : k = 0..2^B−1 }`. The
//! per-dimension MSE drops geometrically with `B`, so recall closes the gap
//! to full-precision search while keeping memory at exactly `B` bits/dim.
//!
//! # What ships
//!
//! - [`rotation::RandomRotation`] — deterministic Haar-uniform orthogonal
//!   matrix (QR of a seeded Gaussian). Reused for build + query.
//! - [`quantize::ExtendedCode`] — packed `B`-bit codes over `D` dimensions,
//!   with `B ∈ {1, 2, 4, 8}` supported end-to-end.
//! - [`index::ExtendedRabitqIndex`] — trait-based [`index::AnnIndex`] impl.
//!   Search is asymmetric: query stays `f32`, candidates are `B`-bit.
//! - [`index::FlatF32Index`] — exact L2 baseline used as ground truth.
//!
//! # Guarantees
//!
//! - Deterministic: `(dim, seed, bits, data)` triple yields bit-identical
//!   codes, index bytes and top-k output across runs and platforms.
//! - No `unsafe`, no external BLAS/LAPACK, no C/C++ dependencies.
//! - File-size discipline: every module is under 500 lines.
//!
//! # Bench
//!
//! `cargo run --release -p ruvector-extended-rabitq --bin erabitq-demo`
//! prints recall\@10 and QPS for `B ∈ {1, 2, 4}` on a synthetic Gaussian
//! corpus. `cargo bench -p ruvector-extended-rabitq` runs distance-kernel
//! micro-benchmarks.

pub mod error;
pub mod index;
pub mod quantize;
pub mod rotation;
pub mod scan;

pub use error::ExtRabitqError;
pub use index::{AnnIndex, ExtendedRabitqIndex, FlatF32Index, SearchResult};
pub use quantize::{ExtendedCode, ExtendedQuantizer};
pub use rotation::RandomRotation;
