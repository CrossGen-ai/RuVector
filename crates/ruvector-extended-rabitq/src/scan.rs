//! Asymmetric distance estimator: f32 query × B-bit database code.
//!
//! Given rotated + unit-normalised query `q̂` (norm `‖q‖`) and rotated +
//! unit-normalised database code `x̂ ≈ decode(code)` (original norm `‖x‖`),
//! we approximate the L2 distance via the classical identity
//!
//! ```text
//!   ‖q − x‖² = ‖q‖² + ‖x‖² − 2 ⟨q, x⟩
//!             = ‖q‖² + ‖x‖² − 2 ‖q‖ ‖x‖ ⟨q̂, x̂⟩
//! ```
//!
//! `⟨q̂, x̂⟩` is estimated as `⟨q̂, decode(code)⟩`. Because the reconstruction
//! error of a symmetric uniform quantizer at `B` bits has mean zero and MSE
//! `Θ(2^{-2B})`, the estimator is unbiased and its variance shrinks
//! geometrically with `B`.
//!
//! Implementation uses a per-query LUT of size `D × 2^B`: for every
//! dimension `d` and every level index `k` we precompute
//! `q̂_d · level[k]`. Scanning a candidate is `D` table lookups + `D` adds.

use crate::quantize::{ExtendedCode, ExtendedQuantizer};

/// Precomputed per-query LUT: shape `dim × n_levels`, row-major.
pub struct QueryLut {
    dim: usize,
    n_levels: usize,
    /// Row-major `dim × n_levels` table.
    table: Vec<f32>,
    /// `‖q‖` for L2 recomposition.
    pub q_norm: f32,
}

impl QueryLut {
    pub fn new(q: &ExtendedQuantizer, q_rot: &[f32]) -> Self {
        assert_eq!(q_rot.len(), q.dim());
        let q_norm_sq: f32 = q_rot.iter().map(|v| v * v).sum();
        let q_norm = q_norm_sq.sqrt();
        let inv = if q_norm > 0.0 { 1.0 / q_norm } else { 0.0 };
        let levels = q.levels();
        let n_levels = levels.len();
        let dim = q.dim();
        let mut table = vec![0f32; dim * n_levels];
        for d in 0..dim {
            let qd = q_rot[d] * inv;
            let base = d * n_levels;
            for k in 0..n_levels {
                table[base + k] = qd * levels[k];
            }
        }
        Self {
            dim,
            n_levels,
            table,
            q_norm,
        }
    }

    /// Estimate `⟨q̂, x̂⟩` from packed code.
    pub fn dot(&self, code: &ExtendedCode, bits: u32) -> f32 {
        let mask = ((1u32 << bits) - 1) as u8;
        let mut acc = 0f32;
        for d in 0..self.dim {
            let bit_off = d * bits as usize;
            let byte_off = bit_off / 8;
            let shift = (bit_off % 8) as u32;
            let idx = ((code.bytes[byte_off] >> shift) & mask) as usize;
            acc += self.table[d * self.n_levels + idx];
        }
        acc
    }

    /// Estimated squared L2 distance to a candidate.
    pub fn l2_sq(&self, code: &ExtendedCode, bits: u32) -> f32 {
        let dot_hat = self.dot(code, bits);
        let x_norm = code.norm;
        // ‖q‖² + ‖x‖² − 2 ‖q‖ ‖x‖ ⟨q̂, x̂⟩
        let est = self.q_norm * self.q_norm + x_norm * x_norm
            - 2.0 * self.q_norm * x_norm * dot_hat;
        est.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::RandomRotation;

    #[test]
    fn scan_close_to_exact_at_high_bits() {
        let dim = 32;
        let rot = RandomRotation::new(dim, 3).unwrap();
        let q = ExtendedQuantizer::new(dim, 8).unwrap();
        let x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.3).sin()).collect();
        let y: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.31 + 0.4).cos()).collect();
        let xr = rot.apply(&x).unwrap();
        let yr = rot.apply(&y).unwrap();
        let code = q.encode(&yr).unwrap();
        let lut = QueryLut::new(&q, &xr);
        let est = lut.l2_sq(&code, 8);
        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let rel_err = (est - exact).abs() / exact.max(1e-6);
        assert!(rel_err < 0.10, "8-bit rel err too high: {rel_err}");
    }
}
