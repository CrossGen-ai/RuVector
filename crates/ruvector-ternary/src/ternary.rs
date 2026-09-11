//! Ternary {-1, 0, +1} encoding with a per-vector magnitude threshold.
//!
//! # Encoding
//!
//! Given a target sparsity `s in (0,1)` (fraction of coordinates that should
//! land in the "0" bucket), we set `theta` to the `s`-quantile of the
//! magnitudes `|x[i]|` of that vector. Coordinates with `|x[i]| <= theta`
//! encode as 0; others as +1 or -1 depending on their sign. This is a
//! *balanced* budget: every vector spends the same number of non-zero
//! coordinates, so codes are directly comparable across the corpus.
//!
//! # Storage
//!
//! Two bitplanes of length `ceil(dim/64)` u64 words:
//!
//! * `sign` — 1 if coordinate is +1, 0 otherwise (including -1 and 0)
//! * `mask` — 1 if coordinate is non-zero
//!
//! Storage per vector = `2 * ceil(dim/64) * 8` bytes = `dim/4` bytes at
//! word-aligned dimensions. That is 2x the memory of plain 1-bit binary but
//! still 16x smaller than fp32 at `dim=128`.
//!
//! # Distance
//!
//! ```text
//!   d_T(a, b) = popcount( (sign_a ^ sign_b) & mask_a & mask_b )
//! ```
//!
//! Counts coordinates where BOTH vectors have declared "meaningful sign" AND
//! that sign disagrees. Small = similar. This is invariant to independent
//! zero-outs on either side, which is exactly what we want: an ambiguous
//! coordinate on either side abstains from voting.

use rand::Rng;

use crate::{words_for, Distance, Encoder};

/// Bitplane-packed ternary code.
#[derive(Clone, Debug)]
pub struct TernaryCode {
    pub sign: Vec<u64>,
    pub mask: Vec<u64>,
    pub dim: usize,
}

impl TernaryCode {
    pub fn bytes(&self) -> usize {
        (self.sign.len() + self.mask.len()) * 8
    }

    /// Number of non-zero coordinates — for debugging / sparsity checks.
    pub fn nnz(&self) -> u32 {
        self.mask.iter().map(|w| w.count_ones()).sum()
    }
}

/// Ternary encoder with a per-vector magnitude threshold.
#[derive(Clone, Copy, Debug)]
pub struct TernaryEncoder {
    pub dim: usize,
    /// Fraction of coordinates to zero out per vector, in [0, 1). 0.0 == all
    /// non-zero (approximately binary with mask=all-ones).
    pub sparsity: f32,
}

impl TernaryEncoder {
    pub fn new(dim: usize, sparsity: f32) -> Self {
        assert!(sparsity >= 0.0 && sparsity < 1.0);
        Self { dim, sparsity }
    }

    /// Return the theta threshold such that a `sparsity` fraction of
    /// coordinates in `v` satisfy `|x| <= theta`.
    fn theta(&self, v: &[f32]) -> f32 {
        if self.sparsity == 0.0 {
            return 0.0;
        }
        let mut mags: Vec<f32> = v.iter().map(|x| x.abs()).collect();
        // deterministic partial sort — use total_cmp for NaN-free ordering
        mags.sort_by(|a, b| a.total_cmp(b));
        let k = ((self.sparsity as f64) * (v.len() as f64)).floor() as usize;
        let k = k.min(v.len().saturating_sub(1));
        mags[k]
    }
}

impl Encoder for TernaryEncoder {
    type Code = TernaryCode;

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim, "dim mismatch");
        let theta = self.theta(v);
        let w = words_for(self.dim);
        let mut sign = vec![0u64; w];
        let mut mask = vec![0u64; w];
        for (i, &x) in v.iter().enumerate() {
            let bit = 1u64 << (i % 64);
            if x.abs() > theta {
                mask[i / 64] |= bit;
                if x > 0.0 {
                    sign[i / 64] |= bit;
                }
            }
        }
        TernaryCode { sign, mask, dim: self.dim }
    }

    fn bytes_per_code(&self) -> usize {
        words_for(self.dim) * 8 * 2
    }

    fn name(&self) -> &'static str {
        "ternary"
    }
}

/// Fused ternary distance.
#[derive(Default, Clone, Copy)]
pub struct TernaryDistance;

impl Distance for TernaryDistance {
    type Code = TernaryCode;

    #[inline]
    fn dist(&self, a: &Self::Code, b: &Self::Code) -> u32 {
        debug_assert_eq!(a.sign.len(), b.sign.len());
        debug_assert_eq!(a.mask.len(), b.mask.len());
        let mut acc: u32 = 0;
        for i in 0..a.sign.len() {
            let disagree = a.sign[i] ^ b.sign[i];
            let both = a.mask[i] & b.mask[i];
            acc += (disagree & both).count_ones();
        }
        acc
    }
}

/// Convenience: encode a corpus in one shot using a seeded, deterministic
/// order.
pub fn encode_corpus<E: Encoder>(enc: &E, vs: &[Vec<f32>]) -> Vec<E::Code> {
    vs.iter().map(|v| enc.encode(v)).collect()
}

/// Deterministic dummy noop that touches `rng` to keep the API stable across
/// tests that want to permute inputs.
pub fn deterministic_shuffle_indices(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    // Fisher–Yates
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        idx.swap(i, j);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seeded_rng;

    #[test]
    fn sparsity_budget_is_respected() {
        // For a dim-1024 vector of iid normals, requesting sparsity=0.5 should
        // zero out ~half the coordinates within a few percent.
        let mut rng = seeded_rng(1);
        use rand_distr::{Distribution, StandardNormal};
        let v: Vec<f32> = (0..1024).map(|_| StandardNormal.sample(&mut rng)).collect();
        let e = TernaryEncoder::new(1024, 0.5);
        let c = e.encode(&v);
        let nnz = c.nnz() as f32 / 1024.0;
        // Expected non-zero fraction is 1 - sparsity = 0.5. Because `theta`
        // is computed by rank and ties are broken deterministically, this is
        // exact to within one coordinate.
        assert!((nnz - 0.5).abs() < 0.01, "nnz frac = {nnz}");
    }

    #[test]
    fn self_distance_is_zero() {
        let mut rng = seeded_rng(7);
        use rand_distr::{Distribution, StandardNormal};
        let v: Vec<f32> = (0..128).map(|_| StandardNormal.sample(&mut rng)).collect();
        let e = TernaryEncoder::new(128, 0.4);
        let c = e.encode(&v);
        let d = TernaryDistance;
        assert_eq!(d.dist(&c, &c), 0);
    }

    #[test]
    fn symmetry() {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = seeded_rng(3);
        let dim = 96;
        let e = TernaryEncoder::new(dim, 0.3);
        let a: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
        let b: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
        let ca = e.encode(&a);
        let cb = e.encode(&b);
        let d = TernaryDistance;
        assert_eq!(d.dist(&ca, &cb), d.dist(&cb, &ca));
    }

    #[test]
    fn abstention_semantics() {
        // Constructed case: two vectors whose signs disagree on a coordinate,
        // but the coordinate is masked out on one side. That coordinate must
        // not contribute to distance.
        let e = TernaryEncoder::new(4, 0.0);
        let mut a = e.encode(&[1.0, 1.0, 1.0, 1.0]);
        let mut b = e.encode(&[-1.0, 1.0, 1.0, 1.0]);
        let d = TernaryDistance;
        assert_eq!(d.dist(&a, &b), 1);
        // Now mask out bit 0 on `a`.
        a.mask[0] &= !1u64;
        assert_eq!(d.dist(&a, &b), 0);
        // Symmetric: mask on `b` instead.
        let mut a = e.encode(&[1.0, 1.0, 1.0, 1.0]);
        let mut b = e.encode(&[-1.0, 1.0, 1.0, 1.0]);
        b.mask[0] &= !1u64;
        assert_eq!(d.dist(&a, &b), 0);
    }
}
