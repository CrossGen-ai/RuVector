//! PQ FastScan with LUT16 (4-bit packed codes + u8 lookup tables).
//!
//! Three trait-swappable Product Quantization (PQ) Asymmetric Distance
//! Computation (ADC) backends compare a query against a corpus of PQ codes:
//!
//! * [`ScalarPqAdc`]  -- baseline PQ-ADC. 1 byte per sub-quantizer (`ks = 256`),
//!   f32 lookup table, straight sum-over-M loop. Jegou 2011.
//! * [`Lut16ScalarFastscan`] -- FastScan: 4-bit codes (`ks = 16`), u8 LUT.
//!   Codes packed 2 per byte. Straight sum-over-M scalar loop.
//! * [`Lut16BlockedFastscan`] -- FastScan with vectors grouped into blocks of
//!   32 (BLOCK = 32). Codes stored in a transposed "PQ column-major" layout:
//!   for each sub-quantizer, 32 4-bit codes live in 16 contiguous bytes,
//!   which is the shape a NEON `TBL` / AVX2 `VPSHUFB` sequence expects.
//!
//! All three backends implement [`PqAdc`], so a caller can swap them at
//! runtime and measure the impact on recall/latency for the same underlying
//! trained codebook.
//!
//! # References
//!
//! * Jegou et al. "Product Quantization for Nearest Neighbor Search" (2011).
//! * Andre et al. "Cache locality is not enough: high-performance nearest
//!   neighbor search with product quantization fast scan" (VLDB 2015).
//! * Guo et al. "Accelerating Large-Scale Inference with Anisotropic Vector
//!   Quantization" (ICML 2020) -- SCANN's LUT16 layout.

#![deny(unsafe_code)]

pub mod codebook;
pub mod fastscan;
pub mod scalar;

pub use codebook::{exact_topk, recall_at_k, sq_l2, SubCodebook};
pub use fastscan::{Lut16BlockedFastscan, Lut16ScalarFastscan};
pub use scalar::ScalarPqAdc;

/// Number of vectors packed together in a FastScan block. 32 * 4 bits =
/// 16 bytes = one NEON `q` register or half an AVX2 `ymm`.
pub const BLOCK: usize = 32;

/// Number of centroids per sub-quantizer for the 4-bit LUT16 variants.
pub const KS16: usize = 16;

/// Error type for training / encoding failures.
#[derive(Debug, thiserror::Error)]
pub enum PqError {
    #[error("dimension {d} not divisible by m={m}")]
    BadShape { d: usize, m: usize },
    #[error("training set empty")]
    EmptyTraining,
    #[error("expected {expected} centroids per sub-quantizer, got {got}")]
    BadKs { expected: usize, got: usize },
}

/// Common interface for a PQ-ADC backend.
pub trait PqAdc {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Bytes of code storage across the whole DB (excluding codebook).
    fn code_bytes(&self) -> usize;
    /// Number of sub-quantizers `m`.
    fn m(&self) -> usize;
    /// Number of centroids per sub-quantizer.
    fn ks(&self) -> usize;
    /// Top-`k` (id, distance) pairs by ascending asymmetric squared-L2.
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>;
}
