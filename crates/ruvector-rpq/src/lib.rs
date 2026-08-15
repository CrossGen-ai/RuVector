//! ruvector-rpq — Residual Product Quantization for high-compression ANN.
//!
//! Provides a common [`Quantizer`] trait and three interchangeable backends:
//! * [`Pq`] — classic single-level Product Quantization (Jégou et al. 2011).
//! * [`Rpq2`] — two-level *Residual* Product Quantization: encode with a
//!   coarse PQ, then encode the residual with a second PQ. Higher recall at
//!   the same coarse-layer bit-budget as [`Pq`].
//! * [`Sq8`] — plain 8-bit scalar quantization (per-dimension uniform), a
//!   compression-only baseline for comparison.
//!
//! All three implement asymmetric distance computation (ADC): a query is
//! kept float, the database is compressed, and squared-L2 distance is
//! computed against the codebook lookup tables at query time.
//!
//! The library is dependency-free, forbids unsafe code, and is
//! deterministic given a seed. See `ADR-306` for design rationale and
//! `docs/research/nightly/2026-08-15-residual-product-quantization/` for
//! the measured benchmark study.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use core::fmt;

pub mod kmeans;
pub mod pq;
pub mod rng;
pub mod rpq2;
pub mod sq8;

pub use kmeans::{kmeans, sq_l2};
pub use pq::Pq;
pub use rng::Rng;
pub use rpq2::{Rpq2, Rpq2Scorer};
pub use sq8::Sq8;

/// Squared-L2 asymmetric-distance quantizer.
///
/// Encoded vectors are opaque byte payloads whose size is reported by
/// [`Quantizer::code_bytes`]. `adc_sq_distance` reconstructs no vectors —
/// it uses precomputed per-subspace lookup tables against the query.
pub trait Quantizer: Send + Sync {
    /// Dimensionality of the input vectors this quantizer expects.
    fn dim(&self) -> usize;
    /// Byte length of an encoded vector.
    fn code_bytes(&self) -> usize;
    /// Encode a single vector. Length of `out` must equal `code_bytes()`.
    fn encode(&self, x: &[f32], out: &mut [u8]);
    /// Compute squared-L2 distance ADC(query, encoded).
    fn adc_sq_distance(&self, query: &[f32], encoded: &[u8]) -> f32;
    /// Human-readable name (used in benchmark output).
    fn name(&self) -> &'static str;
}

/// Compute exact squared-L2 top-`k` neighbours of `query` over `data`.
pub fn brute_topk(query: &[f32], data: &[f32], dim: usize, k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut scored: Vec<(f32, u32)> = (0..n)
        .map(|i| (sq_l2(query, &data[i * dim..(i + 1) * dim]), i as u32))
        .collect();
    scored.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    scored[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Compute recall@k of `predicted` vs `truth`. Both must have length `k`.
pub fn recall_at_k(truth: &[u32], predicted: &[u32]) -> f32 {
    let k = truth.len() as f32;
    let mut hit = 0.0f32;
    for p in predicted {
        if truth.contains(p) {
            hit += 1.0;
        }
    }
    hit / k
}

/// One row of the benchmark output table.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub name: &'static str,
    pub code_bytes: usize,
    pub train_ms: f64,
    pub encode_ms: f64,
    pub query_ms: f64,
    pub recall_at_10: f32,
}

impl fmt::Display for BenchRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<14} | code={:>4}B | train={:>7.1} ms | encode={:>7.1} ms | query={:>7.2} ms/q | recall@10={:.3}",
            self.name,
            self.code_bytes,
            self.train_ms,
            self.encode_ms,
            self.query_ms,
            self.recall_at_10
        )
    }
}
