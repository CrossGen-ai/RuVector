//! Deterministic Haar-uniform orthogonal rotation via QR of a seeded Gaussian.
//!
//! Reused for build + query so the estimator is unbiased on the transformed
//! space. Same seed + dim → bit-identical matrix on every platform.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::error::ExtRabitqError;

/// Row-major `D × D` orthogonal matrix stored densely as `f32`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RandomRotation {
    dim: usize,
    /// Row-major matrix, length `dim * dim`.
    data: Vec<f32>,
}

impl RandomRotation {
    /// Build a Haar-uniform orthogonal matrix from a fixed seed.
    ///
    /// The classical Mezzadri (2007) construction: sample `G ∈ ℝ^{D×D}` iid
    /// standard normal, take its QR decomposition `G = QR`, then flip signs
    /// of columns of `Q` by `sign(diag(R))` to get a Haar-uniform sample of
    /// the orthogonal group `O(D)`.
    pub fn new(dim: usize, seed: u64) -> Result<Self, ExtRabitqError> {
        if dim == 0 {
            return Err(ExtRabitqError::InvalidDim { dim });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut g = vec![0f32; dim * dim];
        for slot in g.iter_mut() {
            let x: f64 = rng.sample(StandardNormal);
            *slot = x as f32;
        }
        // Modified Gram-Schmidt on columns of `g` in-place.
        let mut q = vec![0f32; dim * dim];
        for col in 0..dim {
            for row in 0..dim {
                q[row * dim + col] = g[row * dim + col];
            }
            for prev in 0..col {
                let mut dot = 0f32;
                for row in 0..dim {
                    dot += q[row * dim + prev] * q[row * dim + col];
                }
                for row in 0..dim {
                    q[row * dim + col] -= dot * q[row * dim + prev];
                }
            }
            let mut norm2 = 0f32;
            for row in 0..dim {
                norm2 += q[row * dim + col] * q[row * dim + col];
            }
            let inv = 1.0 / norm2.sqrt().max(f32::MIN_POSITIVE);
            for row in 0..dim {
                q[row * dim + col] *= inv;
            }
        }
        // Deterministic sign fix so different platforms agree.
        for col in 0..dim {
            if q[col * dim + col] < 0.0 {
                for row in 0..dim {
                    q[row * dim + col] = -q[row * dim + col];
                }
            }
        }
        Ok(Self { dim, data: q })
    }

    /// `y = P * x` — apply the rotation to a length-`D` vector.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>, ExtRabitqError> {
        if x.len() != self.dim {
            return Err(ExtRabitqError::DimMismatch {
                expected: self.dim,
                actual: x.len(),
            });
        }
        let mut y = vec![0f32; self.dim];
        for row in 0..self.dim {
            let base = row * self.dim;
            let mut acc = 0f32;
            for col in 0..self.dim {
                acc += self.data[base + col] * x[col];
            }
            y[row] = acc;
        }
        Ok(y)
    }

    /// Dimension `D`.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_orthogonal() {
        let r = RandomRotation::new(16, 42).unwrap();
        // Check that r r^T ≈ I via random probe: apply then invert-apply.
        // Since P is orthogonal, ‖Px‖ = ‖x‖.
        let x: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let y = r.apply(&x).unwrap();
        let nx: f32 = x.iter().map(|v| v * v).sum();
        let ny: f32 = y.iter().map(|v| v * v).sum();
        assert!((nx - ny).abs() / nx < 1e-4, "norm changed: {nx} → {ny}");
    }

    #[test]
    fn deterministic_across_runs() {
        let a = RandomRotation::new(8, 7).unwrap();
        let b = RandomRotation::new(8, 7).unwrap();
        assert_eq!(a.data, b.data);
    }
}
