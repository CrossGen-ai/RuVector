//! PQ FastScan: 4-bit Product Quantization with SIMD shuffle ADC.
//!
//! Provides three measured variants for ANN distance scan:
//!   * `flat`         – brute-force float L2 (ground-truth oracle)
//!   * `pq8`          – standard 8-bit PQ ADC (K=256 per sub-quantizer)
//!   * `fastscan4`    – 4-bit PQ FastScan with NEON `vqtbl1q_u8` shuffle ADC
//!
//! See `docs/research/nightly/2026-05-20-pq-fast-scan/README.md` for theory.

pub mod kmeans;
pub mod pq;
pub mod fastscan;
pub mod flat;

pub use pq::{ProductQuantizer, Pq8Index};
pub use fastscan::{FastScanIndex, FastScanLut};
pub use flat::flat_l2_topk;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PqError {
    #[error("dimension {dim} is not divisible by subquantizer count {m}")]
    BadSubspace { dim: usize, m: usize },
    #[error("expected {expected} vectors, got {actual}")]
    BadVectorCount { expected: usize, actual: usize },
    #[error("training set is empty")]
    EmptyTraining,
    #[error("k={k} exceeds training-set size {n}")]
    TooFewSamples { k: usize, n: usize },
}

pub type Result<T> = std::result::Result<T, PqError>;

/// Squared L2 distance between two equal-length float slices.
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
