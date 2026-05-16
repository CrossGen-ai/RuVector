//! Anisotropic PQ wrapped with a learned orthogonal rotation (OPQ-style).
//!
//! Standard PQ wastes capacity when energy is unevenly distributed across
//! subspaces (e.g. the first few PCA components carry most variance). OPQ
//! (Ge et al., CVPR 2013) learns an orthogonal matrix `R` that minimises
//! reconstruction error after PQ, effectively rotating the data into a basis
//! where subspaces are balanced.
//!
//! In this crate we don't try to *jointly* optimise `R` and the APQ
//! codebooks (the original OPQ paper iterates between them; that path is in
//! `What to improve next`). Instead we use the simpler "parametric OPQ"
//! variant: estimate the data covariance, diagonalise it, and choose a
//! permutation of eigenvectors that **balances** trace energy across
//! subspaces. This is a single closed-form step at train time and gives
//! most of the gain in practice for moderate dimensions.

use crate::apq::Apq;
use crate::{QuantError, Quantizer};

#[derive(Debug, Clone)]
pub struct OpqApq {
    /// Row-major `dim x dim` orthogonal rotation. `x_rotated = R * x`.
    rotation: Vec<f32>,
    dim: usize,
    inner: Apq,
}

impl OpqApq {
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        eta: f32,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self, QuantError> {
        let dim = data[0].len();
        let rotation = learn_balanced_rotation(data, dim, m);
        let rotated: Vec<Vec<f32>> = data.iter().map(|v| apply_rot(&rotation, v, dim)).collect();
        let inner = Apq::train(&rotated, m, k, eta, max_iter, seed)?;
        Ok(Self { rotation, dim, inner })
    }
}

#[inline]
fn apply_rot(r: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    for i in 0..dim {
        let mut s = 0.0f32;
        for j in 0..dim {
            s += r[i * dim + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Build an orthogonal rotation that balances per-subspace variance.
///
/// 1. Compute mean and covariance of `data`.
/// 2. Diagonalise via Jacobi rotations (works fine for dim ~64; we restrict
///    the PoC to moderate dims).
/// 3. Sort eigenvectors by variance descending and assign them round-robin
///    to the `m` subspaces — this equalises trace(Sigma_s) across subspaces.
fn learn_balanced_rotation(data: &[Vec<f32>], dim: usize, m: usize) -> Vec<f32> {
    let n = data.len() as f32;
    let mut mean = vec![0f32; dim];
    for v in data {
        for i in 0..dim { mean[i] += v[i]; }
    }
    for i in 0..dim { mean[i] /= n; }

    // Covariance (dim x dim, row-major).
    let mut cov = vec![0f32; dim * dim];
    for v in data {
        for i in 0..dim {
            let di = v[i] - mean[i];
            for j in 0..dim {
                let dj = v[j] - mean[j];
                cov[i * dim + j] += di * dj;
            }
        }
    }
    for x in &mut cov { *x /= n.max(1.0); }

    // Jacobi eigen-decomposition.
    let mut eig_vec = vec![0f32; dim * dim];
    for i in 0..dim { eig_vec[i * dim + i] = 1.0; }
    jacobi(&mut cov, &mut eig_vec, dim, 50);

    // Read eigenvalues from diagonal.
    let mut eigs: Vec<(usize, f32)> = (0..dim).map(|i| (i, cov[i * dim + i])).collect();
    eigs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Round-robin assignment of sorted eigenvectors to subspaces. Slot `s, k`
    // (i.e. k-th component within subspace s) gets the eigenvector at
    // position `k * m + s` in the sorted list.
    let sub_dim = dim / m;
    let mut rotation = vec![0f32; dim * dim];
    for s in 0..m {
        for kk in 0..sub_dim {
            let pos = kk * m + s;
            if pos >= dim { break; }
            let (eig_idx, _) = eigs[pos];
            let row_in_rot = s * sub_dim + kk;
            for j in 0..dim {
                rotation[row_in_rot * dim + j] = eig_vec[j * dim + eig_idx];
            }
        }
    }
    rotation
}

/// In-place Jacobi diagonalisation. `a` is the symmetric matrix (overwritten
/// with its diagonalised form); `v` accumulates the eigenvectors as columns.
fn jacobi(a: &mut [f32], v: &mut [f32], n: usize, max_sweeps: usize) {
    for _sweep in 0..max_sweeps {
        // Find largest off-diagonal element.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > max_val { max_val = v; p = i; q = j; }
            }
        }
        if max_val < 1e-8 { return; }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Rotate rows/columns p,q of A.
        for i in 0..n {
            let aip = a[i * n + p];
            let aiq = a[i * n + q];
            a[i * n + p] = c * aip - s * aiq;
            a[i * n + q] = s * aip + c * aiq;
        }
        for j in 0..n {
            let apj = a[p * n + j];
            let aqj = a[q * n + j];
            a[p * n + j] = c * apj - s * aqj;
            a[q * n + j] = s * apj + c * aqj;
        }
        // Update eigenvector matrix.
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }
}

impl Quantizer for OpqApq {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let rv = apply_rot(&self.rotation, v, self.dim);
        self.inner.encode(&rv)
    }

    fn decode(&self, code: &[u8]) -> Vec<f32> {
        // Decode in rotated space, then apply R^T (= R^{-1} since orthogonal).
        let rv = self.inner.decode(code);
        let mut out = vec![0f32; self.dim];
        for j in 0..self.dim {
            let mut s = 0.0f32;
            for i in 0..self.dim {
                s += self.rotation[i * self.dim + j] * rv[i];
            }
            out[j] = s;
        }
        out
    }

    fn m(&self) -> usize { self.inner.m() }
    fn k(&self) -> usize { self.inner.k() }

    fn dot_table(&self, q: &[f32]) -> Vec<f32> {
        let rq = apply_rot(&self.rotation, q, self.dim);
        self.inner.dot_table(&rq)
    }
}
