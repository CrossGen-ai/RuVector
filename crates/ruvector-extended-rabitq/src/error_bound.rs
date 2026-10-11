//! Theoretical variance of the eRaBitQ asymmetric distance estimator.
//!
//! For a unit vector u uniformly on the sphere, the rotated coordinates
//! are approximately N(0, 1/D). Each coordinate is quantized with
//! codebook MSE σ²_B (in standard-normal units). The estimator for
//! <q, x> satisfies E[error] = 0 and
//!   Var(<q, u_hat> - <q, u>) ≤ ||q||^2 * σ²_B / D
//! so the IP estimator variance is
//!   Var(<q, x>_hat - <q, x>) ≤ ||q||^2 * ||x||^2 * σ²_B / D.

use crate::codebook::Codebook;

#[derive(Clone, Debug)]
pub struct VarianceBound {
    pub bits: u8,
    pub dim: usize,
    pub sigma_sq_per_coord: f32,
    pub normalised_variance: f32,
}

impl VarianceBound {
    pub fn new(bits: u8, dim: usize) -> Self {
        let cb = Codebook::gaussian(bits);
        let s2 = cb.mse_gaussian();
        Self {
            bits,
            dim,
            sigma_sq_per_coord: s2,
            normalised_variance: s2 / (dim as f32),
        }
    }

    /// Upper bound on Var(estimator) for a given query and vector norm.
    pub fn ip_variance_bound(&self, q_norm: f32, x_norm: f32) -> f32 {
        q_norm * q_norm * x_norm * x_norm * self.normalised_variance
    }

    /// Approximate 2sigma relative error for inner product with unit vectors.
    pub fn two_sigma_rel_ip(&self) -> f32 {
        2.0 * self.normalised_variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_scales_with_bits_and_dim() {
        let a = VarianceBound::new(1, 128);
        let b = VarianceBound::new(4, 128);
        let c = VarianceBound::new(4, 1024);
        assert!(a.normalised_variance > b.normalised_variance);
        assert!(b.normalised_variance > c.normalised_variance);
    }
}
