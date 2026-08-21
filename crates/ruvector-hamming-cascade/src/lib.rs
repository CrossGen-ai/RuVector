//! ruvector-hamming-cascade
//!
//! A cascade approximate-nearest-neighbor (ANN) retrieval path built around a
//! swappable [`DistanceOracle`] trait. It ships three concrete oracles that
//! trade recall for latency and memory bandwidth:
//!
//! * [`Fp32Oracle`]  — exact L2 distance on the raw float vectors (baseline)
//! * [`Int8Oracle`]  — per-vector min/max scalar quantization, INT8 L2 SAD
//! * [`HammingOracle`] — 1-bit sign quantization, POPCNT-based Hamming filter
//!
//! The [`Cascade`] index chains oracles: a fast/lossy oracle produces a
//! shortlist of size `probe_k`, which is then re-ranked with the exact FP32
//! oracle to recover recall. This is the same pattern used in Faiss binary
//! indexes, Milvus `BIN_IVF_FLAT + FLOAT_RERANK`, and the Qdrant Binary
//! Quantization + oversampling story, generalized behind a stable Rust
//! interface with real benchmarks (`cargo run --release --bin cascade-report`).
//!
//! All storage is column-major-free contiguous `Vec<f32>` / `Vec<u8>` /
//! `Vec<u64>` for cache-friendly scans. No SIMD intrinsics are used — the
//! goal is to show what portable, safe Rust already buys.

#![forbid(unsafe_code)]

pub mod cascade;
pub mod oracle;
pub mod quantize;

pub use cascade::{Cascade, CascadeConfig, Hit};
pub use oracle::{DistanceOracle, Fp32Oracle, HammingOracle, Int8Oracle};

/// Semantic version of the on-disk / on-wire layout produced by this crate.
/// Bump when the codebook layout of any oracle changes.
pub const LAYOUT_VERSION: u32 = 1;
