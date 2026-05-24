//! ruvector-opq — Optimized Product Quantization.
//!
//! Three quantizers expose the same [`Quantizer`] trait so callers can swap
//! backends at runtime:
//!
//! * [`pq::Pq`]        — classic Product Quantization (Jégou 2011).
//! * [`opq::OpqNp`]    — Non-Parametric OPQ via eigenvalue allocation
//!                       (Ge 2013 §4.1) — closed form, no iteration.
//! * [`opq::OpqP`]     — Parametric OPQ via Orthogonal Procrustes
//!                       (Ge 2013 §4.2) — iterative, slower, lower error.
//!
//! All three encode a `d`-vector into `m` bytes (`k=256` centroids/sub).

pub mod kmeans;
pub mod pq;
pub mod opq;
pub mod recall;

pub use pq::Pq;
pub use opq::{OpqNp, OpqP};

/// Common interface for PQ-family quantizers.
pub trait Quantizer {
    /// Train on `n × d` row-major data.
    fn fit(&mut self, data: &[f32], n: usize, d: usize);
    /// Encode a single `d`-vector to `m` bytes.
    fn encode(&self, x: &[f32], out: &mut [u8]);
    /// Reconstruct from `m`-byte code into a `d`-vector.
    fn decode(&self, code: &[u8], out: &mut [f32]);
    /// Asymmetric distance: query (uncompressed) vs encoded code.
    /// Returns squared L2 distance.
    fn adc(&self, query: &[f32], code: &[u8]) -> f32;
    /// Build a per-query lookup table: `m × 256` row-major squared L2 from
    /// each subspace of the (possibly rotated) query to each centroid.
    /// Production hot loop calls this once per query then uses [`adc_lut`].
    fn build_lut(&self, query: &[f32], lut: &mut [f32]);
    /// Sum the lookup table along the m subspaces using the byte code.
    /// `lut` must have been built by [`build_lut`] for this query.
    fn adc_lut(&self, lut: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(lut.len(), self.m() * 256);
        debug_assert_eq!(code.len(), self.m());
        let mut s = 0.0f32;
        for sidx in 0..self.m() {
            s += lut[sidx * 256 + code[sidx] as usize];
        }
        s
    }
    fn m(&self) -> usize;
    fn d(&self) -> usize;
}

/// Squared L2 between two equal-length slices.
#[inline]
pub fn sql2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Mean squared reconstruction error over `n` rows of dimension `d`.
pub fn mse(orig: &[f32], recon: &[f32], n: usize, d: usize) -> f32 {
    assert_eq!(orig.len(), n * d);
    assert_eq!(recon.len(), n * d);
    let mut s = 0.0f64;
    for i in 0..n {
        for j in 0..d {
            let e = (orig[i * d + j] - recon[i * d + j]) as f64;
            s += e * e;
        }
    }
    (s / (n * d) as f64) as f32
}
