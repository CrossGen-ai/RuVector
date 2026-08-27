//! ruvector-avq — Anisotropic Vector Quantization (AVQ) codebooks for
//! inner-product ANN.
//!
//! Implements three variants sharing the same [`Quantizer`] trait:
//!
//! * [`pq::PqMse`] — plain product quantization trained by MSE k-means
//!   (baseline).
//! * [`avq::AvqScoreAware`] — PQ codebooks trained with the score-aware
//!   loss `h_∥ ||e_∥||² + h_⊥ ||e_⊥||²` (Guo et al., ICML 2020).
//! * [`avq_norm::AvqNorm`] — AVQ + per-vector norm rescaling. Vectors are
//!   unit-normalised for codebook training and a per-vector norm is
//!   stored on the side; ADC scores are rescaled at query time.
//!
//! Distance model is inner-product (larger = closer). All three
//! variants share the same Asymmetric Distance Computation (ADC) LUT
//! backend, so any measured recall difference is attributable to the
//! codebook, not the search kernel.

pub mod data;
pub mod kmeans;
pub mod avq_kmeans;
pub mod pq;
pub mod avq;
pub mod avq_norm;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors surfaced by the crate.
#[derive(Debug, Error)]
pub enum AvqError {
    #[error("dimension {dim} is not divisible by subspace count {m}")]
    BadSubspaces { dim: usize, m: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("k-means asked for {k} centroids from {n} points")]
    NotEnoughPoints { k: usize, n: usize },
    #[error("shape mismatch: expected dim {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
}

/// Configuration common to every quantizer in this crate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantizerConfig {
    /// Number of subspaces (`M`). Must divide `dim`.
    pub m: usize,
    /// Codebook size per subspace (`K_s`). Typically 256 so each code
    /// fits in one byte.
    pub ks: usize,
    /// K-means iterations per subspace.
    pub iters: usize,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for QuantizerConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ks: 256,
            iters: 12,
            seed: 0xA5EED_A5EED_u64,
        }
    }
}

/// Anisotropic-loss knob used by [`avq::AvqScoreAware`] and
/// [`avq_norm::AvqNorm`]. `t` is the target inner-product threshold
/// from the original paper; larger `t` puts more weight on parallel
/// (score) error.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnisotropicConfig {
    pub t: f32,
}

impl Default for AnisotropicConfig {
    fn default() -> Self {
        Self { t: 0.2 }
    }
}

impl AnisotropicConfig {
    /// η = (d-1) · T² / (1 − T²) — parallel-vs-orthogonal weight ratio
    /// derived from the target inner-product threshold (Guo et al.,
    /// §4.2). Falls back to 1.0 when `t` degenerates.
    pub fn eta(&self, dim: usize) -> f32 {
        let t2 = self.t * self.t;
        if t2 >= 1.0 || dim < 2 {
            1.0
        } else {
            (dim as f32 - 1.0) * t2 / (1.0 - t2)
        }
    }
}

/// Common quantizer trait. Every variant is train-once / encode-once /
/// query-many. `adc` returns one inner-product score per code.
pub trait Quantizer {
    fn train(&mut self, data: &[Vec<f32>]) -> Result<(), AvqError>;
    fn encode(&self, data: &[Vec<f32>]) -> Result<Vec<u8>, AvqError>;
    fn adc(&self, query: &[f32], codes: &[u8]) -> Result<Vec<f32>, AvqError>;
    /// Bytes stored per vector (encoded footprint only, excludes
    /// codebook / side-channel).
    fn code_bytes(&self) -> usize;
    /// Optional per-vector side channel bytes (e.g. AvqNorm stores 4).
    fn side_bytes(&self) -> usize {
        0
    }
}

/// Convenience wrapper: dot product.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Ground-truth top-K inner product (brute force, for eval only).
pub fn brute_top_k(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, v)| (i, dot(query, v))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Recall@k of `predicted` vs `truth`.
pub fn recall_at_k(predicted: &[usize], truth: &[usize]) -> f32 {
    if truth.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for t in truth {
        if predicted.contains(t) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

/// Top-K on a slice of ADC scores.
pub fn top_k_scores(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    idx.into_iter().take(k).collect()
}
