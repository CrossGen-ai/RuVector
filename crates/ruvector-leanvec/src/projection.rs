//! Learned linear projection (PCA via power iteration on the covariance).
//!
//! We avoid any LAPACK dependency: PCA components are extracted one at a time
//! by power iteration on `C = (1/n) Σ (x − μ)(x − μ)^T`, with explicit
//! Gram-Schmidt deflation against previously-extracted components. This is
//! O(iter · d · n) per component — fine for the moderate `r ≪ d` regime
//! LeanVec targets (r = d/2 or d/4).
//!
//! The projection is stored row-major as an `r × d` matrix `p`. Projecting a
//! vector `v ∈ R^d` produces `Pv ∈ R^r`. Components are forced orthonormal so
//! the projection preserves L2 distances up to the truncation error from the
//! discarded `d − r` directions — the same approximation Tepper et al. (2024)
//! use as the LeanVec "secondary" geometry.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// An `r × d` row-major orthonormal projection.
#[derive(Clone, Debug)]
pub struct Projection {
    /// Input dimensionality.
    pub d: usize,
    /// Output (reduced) dimensionality.
    pub r: usize,
    /// Row-major projection matrix, length `r * d`.
    pub p: Vec<f32>,
    /// Per-dimension mean of the training set; subtracted before projecting.
    pub mean: Vec<f32>,
}

impl Projection {
    /// Identity projection: keeps all `d` dimensions, zero mean. Useful as a
    /// pass-through baseline so LVQ-only and LeanVec+LVQ share one code path.
    pub fn identity(d: usize) -> Self {
        let mut p = vec![0.0_f32; d * d];
        for i in 0..d {
            p[i * d + i] = 1.0;
        }
        Self { d, r: d, p, mean: vec![0.0; d] }
    }

    /// Train an orthonormal projection from a row-major `[n × d]` sample.
    ///
    /// `r` must satisfy `0 < r ≤ d`. Uses `iters` power iterations per
    /// component (default 30 is plenty for normalised data).
    pub fn train_pca(samples: &[f32], n: usize, d: usize, r: usize, seed: u64) -> Self {
        assert!(r > 0 && r <= d, "r must be in (0, d]");
        assert_eq!(samples.len(), n * d, "samples length must equal n*d");

        let mut mean = vec![0.0_f32; d];
        for i in 0..n {
            let row = &samples[i * d..(i + 1) * d];
            for j in 0..d {
                mean[j] += row[j];
            }
        }
        let inv_n = 1.0 / n as f32;
        for j in 0..d {
            mean[j] *= inv_n;
        }

        // Centered copy. n×d floats; acceptable for the training-sample sizes
        // LeanVec actually uses (10k–100k vectors).
        let mut centered = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                centered[i * d + j] = samples[i * d + j] - mean[j];
            }
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut p = vec![0.0_f32; r * d];

        for k in 0..r {
            // Random unit start vector.
            let mut v = vec![0.0_f32; d];
            for j in 0..d {
                v[j] = rng.gen::<f32>() - 0.5;
            }
            // Deflate the start vector against earlier components so power
            // iteration converges to the next principal direction, not back to
            // an earlier one.
            for prev in 0..k {
                let dot = dot(&v, &p[prev * d..(prev + 1) * d]);
                axpy(-dot, &p[prev * d..(prev + 1) * d], &mut v);
            }
            normalize(&mut v);

            for _ in 0..30 {
                // u = (1/n) Σ x_i (x_i · v)  — covariance × v
                let mut u = vec![0.0_f32; d];
                for i in 0..n {
                    let row = &centered[i * d..(i + 1) * d];
                    let s = dot(row, &v);
                    axpy(s * inv_n, row, &mut u);
                }
                for prev in 0..k {
                    let dot_u = dot(&u, &p[prev * d..(prev + 1) * d]);
                    axpy(-dot_u, &p[prev * d..(prev + 1) * d], &mut u);
                }
                normalize(&mut u);
                v = u;
            }

            p[k * d..(k + 1) * d].copy_from_slice(&v);
        }

        Self { d, r, p, mean }
    }

    /// Project `v ∈ R^d` to `out ∈ R^r`. `out.len()` must equal `self.r`.
    pub fn project_into(&self, v: &[f32], out: &mut [f32]) {
        debug_assert_eq!(v.len(), self.d);
        debug_assert_eq!(out.len(), self.r);
        for k in 0..self.r {
            let row = &self.p[k * self.d..(k + 1) * self.d];
            let mut s = 0.0_f32;
            for j in 0..self.d {
                s += row[j] * (v[j] - self.mean[j]);
            }
            out[k] = s;
        }
    }

    /// Convenience: allocates the output vector.
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.r];
        self.project_into(v, &mut out);
        out
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    let inv = 1.0 / n;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn identity_is_passthrough() {
        let p = Projection::identity(4);
        let v = vec![1.0, -2.0, 3.0, 0.5];
        let out = p.project(&v);
        for i in 0..4 {
            assert!((out[i] - v[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn pca_components_orthonormal() {
        // Synthetic data with two dominant axes of variance.
        let n = 400;
        let d = 8;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            let a = rng.gen::<f32>() - 0.5;
            let b = rng.gen::<f32>() - 0.5;
            data[i * d + 0] = 5.0 * a;
            data[i * d + 1] = 5.0 * a + 0.1 * b;
            data[i * d + 2] = 3.0 * b;
            for j in 3..d {
                data[i * d + j] = 0.01 * (rng.gen::<f32>() - 0.5);
            }
        }
        let proj = Projection::train_pca(&data, n, d, 3, 11);
        assert_eq!(proj.r, 3);
        // Orthonormality: P P^T ≈ I_r.
        for a in 0..3 {
            for b in 0..3 {
                let mut s = 0.0;
                for j in 0..d {
                    s += proj.p[a * d + j] * proj.p[b * d + j];
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-3, "P P^T[{},{}] = {} (want {})", a, b, s, expected);
            }
        }
    }
}
