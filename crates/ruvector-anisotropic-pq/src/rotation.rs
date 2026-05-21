//! Orthogonal rotation utilities for OPQ.
//!
//! We build an orthogonal `R` (d x d) by composing random Givens rotations
//! drawn from a seeded RNG, then learn it greedily by alternating:
//!   1. fix R, train per-subspace codebooks on rotated data
//!   2. fix codebooks, update R via the orthogonal-Procrustes solution
//!      using a power-iteration SVD on small d (we keep d modest in tests).
//!
//! For benchmark realism we use the **non-parametric OPQ** approach
//! (Procrustes update) rather than the parametric variant. SVD on dxd is
//! done via Jacobi rotations — exact, O(d^3), fine for d up to a few
//! hundred which matches the experimental setup in the OPQ paper.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub type Matrix = Vec<Vec<f32>>; // row-major, d x d

pub fn identity(d: usize) -> Matrix {
    let mut m = vec![vec![0f32; d]; d];
    for i in 0..d {
        m[i][i] = 1.0;
    }
    m
}

pub fn random_orthogonal(d: usize, seed: u64) -> Matrix {
    // Apply ~ d*log(d) random Givens rotations to identity to get a
    // sample from O(d) without using a heavy linear algebra crate.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = identity(d);
    let steps = d * ((d as f32).ln().ceil() as usize + 1);
    for _ in 0..steps {
        let i = rng.gen_range(0..d);
        let mut j = rng.gen_range(0..d);
        if j == i {
            j = (j + 1) % d;
        }
        let theta = rng.gen::<f32>() * std::f32::consts::TAU;
        let c = theta.cos();
        let s = theta.sin();
        for r in 0..d {
            let a = m[r][i];
            let b = m[r][j];
            m[r][i] = c * a - s * b;
            m[r][j] = s * a + c * b;
        }
    }
    m
}

#[inline]
pub fn rotate(r: &Matrix, x: &[f32], out: &mut [f32]) {
    let d = x.len();
    debug_assert_eq!(r.len(), d);
    debug_assert_eq!(out.len(), d);
    for j in 0..d {
        let mut s = 0f32;
        let row = &r[j];
        for i in 0..d {
            s += row[i] * x[i];
        }
        out[j] = s;
    }
}

pub fn rotate_vec(r: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    rotate(r, x, &mut out);
    out
}

/// Matrix multiply: C = A * B where all are d x d, row-major.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let d = a.len();
    let mut c = vec![vec![0f32; d]; d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..d {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

pub fn transpose(a: &Matrix) -> Matrix {
    let d = a.len();
    let mut t = vec![vec![0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Jacobi-based symmetric eigendecomposition for a d x d symmetric matrix.
/// Returns (eigenvalues, eigenvectors-as-columns-of-V). O(d^3) and accurate
/// for small/moderate d.
pub fn jacobi_eig(a_in: &Matrix) -> (Vec<f32>, Matrix) {
    let d = a_in.len();
    let mut a = a_in.clone();
    let mut v = identity(d);
    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        // off-diagonal magnitude
        let mut off = 0f32;
        for p in 0..d {
            for q in (p + 1)..d {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-7 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p][q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // update rows/cols p,q
                for r in 0..d {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = c * arp - s * arq;
                    a[r][q] = s * arp + c * arq;
                }
                for r in 0..d {
                    let apr = a[p][r];
                    let aqr = a[q][r];
                    a[p][r] = c * apr - s * aqr;
                    a[q][r] = s * apr + c * aqr;
                }
                // accumulate V
                for r in 0..d {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let eigs: Vec<f32> = (0..d).map(|i| a[i][i]).collect();
    (eigs, v)
}

/// Solve orthogonal Procrustes: given X (n x d) and Y (n x d), find orthogonal
/// R (d x d) minimizing ||X R - Y||_F. R = U V^T from SVD of X^T Y.
///
/// We compute SVD via the symmetric eigendecomposition of (X^T Y)^T (X^T Y).
pub fn procrustes(xty: &Matrix) -> Matrix {
    let d = xty.len();
    // M = (X^T Y)^T (X^T Y)  → V holds right singular vectors
    let mtm = matmul(&transpose(xty), xty);
    let (_eigs, v) = jacobi_eig(&mtm);
    // U sigma = (X^T Y) V  → U = (X^T Y) V / sigma
    let xy_v = matmul(xty, &v);
    let mut u = vec![vec![0f32; d]; d];
    for j in 0..d {
        let mut norm = 0f32;
        for i in 0..d {
            norm += xy_v[i][j] * xy_v[i][j];
        }
        let norm = norm.sqrt().max(1e-12);
        for i in 0..d {
            u[i][j] = xy_v[i][j] / norm;
        }
    }
    // R = U V^T
    matmul(&u, &transpose(&v))
}
