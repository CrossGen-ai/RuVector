//! Variance-balancing orthonormal rotation for PQ (non-parametric OPQ).
//!
//! We compute the eigenbasis of the (centered) data covariance via a simple
//! Jacobi eigendecomposition — no BLAS, no `ndarray`. The eigen-directions
//! are then *permuted* so that the sum of eigenvalues per PQ subspace is
//! roughly equal. That is enough to reproduce the "variance balancing" that
//! non-parametric OPQ (Ge et al., 2013) achieves, giving APQ a fair chance
//! per subspace.
//!
//! For high `dim` (>128) a Jacobi sweep gets expensive (O(dim^3) per sweep),
//! but the nightly PoC targets dim≤128 where Jacobi is still very fast
//! (~milliseconds on modern hardware).

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Column-major orthonormal rotation matrix (`dim × dim`).
pub struct Rotation {
    dim: usize,
    /// Row-major: `mat[i*dim + j]` is entry (i, j).
    mat: Vec<f32>,
}

impl Rotation {
    /// Fit a rotation that balances per-subspace variance for `m` PQ subspaces.
    pub fn fit_variance_balancing(data: &[Vec<f32>], m: usize, seed: u64) -> Self {
        let dim = data[0].len();
        assert!(dim % m == 0);
        let ds = dim / m;

        // Mean.
        let mut mean = vec![0.0f32; dim];
        for x in data {
            for j in 0..dim {
                mean[j] += x[j];
            }
        }
        let inv_n = 1.0 / data.len() as f32;
        for j in 0..dim {
            mean[j] *= inv_n;
        }

        // Covariance (symmetric).
        let mut cov = vec![0.0f32; dim * dim];
        for x in data {
            for i in 0..dim {
                let a = x[i] - mean[i];
                for j in i..dim {
                    let b = x[j] - mean[j];
                    cov[i * dim + j] += a * b;
                }
            }
        }
        for i in 0..dim {
            for j in i..dim {
                cov[i * dim + j] *= inv_n;
                cov[j * dim + i] = cov[i * dim + j];
            }
        }

        // Jacobi eigendecomposition — small dims only.
        let (eig, vecs) = jacobi(&cov, dim, 64, 1e-6);

        // Sort eigenpairs descending.
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| eig[b].partial_cmp(&eig[a]).unwrap());

        // Greedy balance: distribute sorted eigenvectors round-robin into `m`
        // bins so bin variance is roughly equal. Assign bins to subspace
        // slots left-to-right, filling each slot with `ds` eigenvectors.
        let mut bins: Vec<Vec<usize>> = (0..m).map(|_| Vec::with_capacity(ds)).collect();
        let mut bin_sum = vec![0.0f32; m];
        let mut rng = StdRng::seed_from_u64(seed);
        let _ = &mut rng; // seeded for tie-break determinism if we extend later
        for &idx in &order {
            // Fill the least-loaded bin that still has room.
            let mut target = 0usize;
            let mut best = f32::INFINITY;
            for b in 0..m {
                if bins[b].len() < ds && bin_sum[b] < best {
                    best = bin_sum[b];
                    target = b;
                }
            }
            bin_sum[target] += eig[idx];
            bins[target].push(idx);
        }

        // Compose rotation: rows are the permuted eigenvectors (columns of `vecs`).
        let mut mat = vec![0.0f32; dim * dim];
        let mut row = 0usize;
        for bin in &bins {
            for &c in bin {
                for r in 0..dim {
                    mat[row * dim + r] = vecs[r * dim + c];
                }
                row += 1;
            }
        }
        Self { dim, mat }
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += self.mat[i * d + j] * x[j];
            }
            out[i] = acc;
        }
        out
    }

    /// R is orthonormal so R^{-1} = R^T.
    pub fn apply_inverse(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let d = self.dim;
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += self.mat[j * d + i] * x[j];
            }
            out[i] = acc;
        }
        out
    }
}

/// Symmetric-matrix Jacobi eigendecomposition.
/// Returns (eigenvalues, eigenvectors_row_major_column_stored).
///
/// `vecs[r * dim + c]` is the r-th coordinate of the c-th eigenvector.
fn jacobi(input: &[f32], dim: usize, max_sweeps: usize, tol: f32) -> (Vec<f32>, Vec<f32>) {
    let mut a = input.to_vec();
    let mut v = vec![0.0f32; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut off = 0.0f32;
        for p in 0..dim {
            for q in (p + 1)..dim {
                off += a[p * dim + q].abs();
            }
        }
        if off < tol {
            break;
        }
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = a[p * dim + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let app = a[p * dim + p];
                let aqq = a[q * dim + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update A
                for r in 0..dim {
                    let arp = a[r * dim + p];
                    let arq = a[r * dim + q];
                    a[r * dim + p] = c * arp - s * arq;
                    a[r * dim + q] = s * arp + c * arq;
                }
                for r in 0..dim {
                    let apr = a[p * dim + r];
                    let aqr = a[q * dim + r];
                    a[p * dim + r] = c * apr - s * aqr;
                    a[q * dim + r] = s * apr + c * aqr;
                }
                a[p * dim + q] = 0.0;
                a[q * dim + p] = 0.0;
                // Update V
                for r in 0..dim {
                    let vrp = v[r * dim + p];
                    let vrq = v[r * dim + q];
                    v[r * dim + p] = c * vrp - s * vrq;
                    v[r * dim + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let eig: Vec<f32> = (0..dim).map(|i| a[i * dim + i]).collect();
    (eig, v)
}
