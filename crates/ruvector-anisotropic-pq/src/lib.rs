//! Anisotropic Product Quantization for Maximum Inner Product Search (MIPS).
//!
//! Standard PQ trains sub-space codebooks by minimizing the *isotropic*
//! reconstruction error E‖x − q(x)‖². For MIPS the quantity that actually
//! matters is the inner-product residual q^T(x − q(x)); ScaNN (Guo et al.,
//! ICML 2020) showed that penalizing residual components **parallel** to the
//! vector itself much more heavily than orthogonal ones dramatically improves
//! top-k recall at fixed code size — because parallel residuals bias the
//! score, while orthogonal ones average out.
//!
//! This crate implements a compact, dependency-free anisotropic PQ trainer,
//! an ADC (asymmetric distance computation) MIPS index, and a swappable
//! [`PqCodebookTrainer`] trait so isotropic and anisotropic training can be
//! compared under identical harness conditions.
//!
//! Design goals (matches sibling crates like `ruvector-pq-search`):
//! * Safe Rust, no unsafe, no BLAS, no allocator tricks.
//! * Sub-vector dimension ≤ 32 so the per-centroid dxd Gram solve stays cheap.
//! * All numbers reported by the bench binary are measured, not simulated.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod codebook;
pub mod index;

pub use codebook::{
    AnisotropicTrainer, IsotropicTrainer, PqCodebook, PqCodebookTrainer, PqTrainConfig,
};
pub use index::{AnisoPqIndex, MipsResult};

/// Errors surfaced by the crate.
#[derive(Debug, thiserror::Error)]
pub enum PqError {
    /// The requested sub-vector dimension does not divide the vector dim.
    #[error("dim {dim} is not divisible by M={m}")]
    BadShape {
        /// Full vector dimension.
        dim: usize,
        /// Number of sub-spaces.
        m: usize,
    },
    /// Codebook was trained on a different shape than the encode/search input.
    #[error("shape mismatch: index is {expected}-d, query is {got}-d")]
    ShapeMismatch {
        /// Dimension the index was built for.
        expected: usize,
        /// Dimension of the offending input.
        got: usize,
    },
    /// The training set is too small for the requested K.
    #[error("training set of {n} vectors is smaller than K={k}")]
    TooFewSamples {
        /// Number of samples supplied.
        n: usize,
        /// Requested codebook size.
        k: usize,
    },
}

/// Compute the recall@k of an approximate top-k against a ground-truth top-k.
///
/// `approx` and `truth` are both id lists (already truncated / ranked); the
/// result is |approx ∩ truth| / k, using the smaller of the two if either is
/// shorter than `k`.
pub fn recall_at_k(approx: &[usize], truth: &[usize], k: usize) -> f32 {
    let k = k.min(approx.len()).min(truth.len());
    if k == 0 {
        return 0.0;
    }
    let set: std::collections::HashSet<usize> = approx.iter().take(k).copied().collect();
    let hits = truth.iter().take(k).filter(|id| set.contains(id)).count();
    hits as f32 / k as f32
}

/// Inner product of two equal-length slices; panics on length mismatch.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Squared L2 norm.
#[inline]
pub(crate) fn norm_sq(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Squared L2 distance between two equal-length slices.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Return the argmax of a slice of scores. Returns 0 for empty input.
#[inline]
pub(crate) fn argmax(scores: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in scores.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
