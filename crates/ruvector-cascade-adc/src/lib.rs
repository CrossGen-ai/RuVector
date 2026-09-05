//! Cascade Asymmetric Distance Computation (Cascade-ADC).
//!
//! Two-stage progressive-precision PQ scan:
//!   Stage 1: coarse 4-bit sub-quantizer distances (packed 2 codes/byte, tiny LUT
//!            that fits in L1 cache) — used to prune a large candidate list
//!            (typically N) down to a small candidate set (top T ≈ ρN).
//!   Stage 2: fine 8-bit sub-quantizer distances on the surviving T candidates
//!            for accurate ranking of the final top-K.
//!
//! The two stages share a common training set of vectors: 4-bit codebook is
//! derived from the 8-bit codebook by k-means clustering the 256 centroids
//! down to 16 centroids per subspace. This preserves a coherent embedding
//! so a candidate whose 4-bit distance is small tends to also have a small
//! 8-bit distance.
//!
//! The crate provides three concrete scanners implementing the same
//! [`Scanner`] trait, so a caller can benchmark or A/B against each other:
//!
//!   * [`FullEightBitScanner`] — baseline, all N codes at 8-bit precision.
//!   * [`FullFourBitScanner`]  — aggressive, all N codes at 4-bit precision.
//!   * [`CascadeScanner`]      — 4-bit prune + 8-bit refine on top rho fraction.
//!
//! Design goals:
//!   * Deterministic: seeded RNG in tests/bench so runs are reproducible.
//!   * No unsafe, no SIMD intrinsics — vanilla loops that the compiler
//!     auto-vectorises. This crate is about the *algorithmic* saving, not
//!     hand-tuned SIMD (which is orthogonal and can be layered on later).
//!   * Zero allocations in the inner scan (working buffers are re-usable).

#![forbid(unsafe_code)]

pub mod codebook;
pub mod pq;
pub mod scan;

pub use codebook::{Codebook4, Codebook8, TrainingConfig};
pub use pq::{PqIndex, PqParams};
pub use scan::{CascadeScanner, FullEightBitScanner, FullFourBitScanner, ScanResult, Scanner};

/// Estimated bytes-per-vector for a PQ code layout.
///
/// * 8-bit PQ: `m` bytes per vector.
/// * 4-bit PQ: `ceil(m / 2)` bytes per vector.
/// * Cascade:  both layouts held in memory, so `m + ceil(m/2)` bytes.
pub fn bytes_per_vector(m: usize, layout: Layout) -> usize {
    match layout {
        Layout::EightBit => m,
        Layout::FourBit => m.div_ceil(2),
        Layout::Cascade => m + m.div_ceil(2),
    }
}

/// Storage layout used by a scanner.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Layout {
    EightBit,
    FourBit,
    Cascade,
}
