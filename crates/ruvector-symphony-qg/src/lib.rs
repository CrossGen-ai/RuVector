//! ruvector-symphony-qg
//!
//! Symphonious integration of 1-bit quantization and graph-based ANN search.
//!
//! Implements three variants for benchmark comparison:
//!   1. `ExactGraph`         – greedy graph search with exact f32 L2 distances (baseline).
//!   2. `SymphonyQG`         – quantized first-pass distance estimates + float re-rank.
//!   3. `SymphonyQGPacked`   – same as (2) but with unrolled popcount over u64 words
//!                              and cache-aligned neighbor blocks for fast traversal.
//!
//! Inspired by SymphonyQG (SIGMOD 2025): unify quantization with the graph traversal
//! so the dominant cost (distance computations during graph exploration) is paid in
//! the quantized space, while the final answers are decided by exact distances on a
//! tiny re-rank set.
#![allow(clippy::needless_range_loop)]

pub mod quantize;
pub mod graph;
pub mod symphony;

pub use quantize::{RotatedQuantizer, QuantizedCode};
pub use graph::{ExactGraph, GraphParams};
pub use symphony::{SymphonyQG, SymphonyQGPacked, SearchStats};
