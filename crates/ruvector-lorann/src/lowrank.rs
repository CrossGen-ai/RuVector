//! Per-cluster reduced-rank regression — the LoRANN core.
//!
//! For a cluster of `n_c` points `X_c ∈ R^{n_c × d}` we want an orthonormal
//! basis `V_c ∈ R^{d × r}` such that the top-`r` directions of `X_c^T X_c`
//! are captured. We compute it with subspace iteration on `M = X_c^T X_c`
//! (a `d × d` matrix) — robust and dependency-free for the sizes we target.
//!
//! Given `V_c`, the reduced coordinates are `A_c = X_c V_c ∈ R^{n_c × r}` and
//! a query `q ∈ R^d` is scored against the cluster as
//!
//! ```text
//! score_c(q, i) = A_c[i] · (q^T V_c)        // O(r) per point
//! ```
//!
//! versus exact `O(d)` per point. Approximate top-`m` per cluster are then
//! rescored with the exact dot product against the original `X_c` rows.
//!
//! This file deliberately keeps the linear algebra elementary (Gram matrix +
//! Gram–Schmidt + power iteration). For larger `d` swap in BLAS / `ndarray`.

use crate::Lcg;

/// Compute `M = X^T X` for row-major `X` of shape `[n × d]`. Output is `[d × d]`.
pub fn gram(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n * d);
    let mut m = vec![0.0_f32; d * d];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        for a in 0..d {
            let xa = row[a];
            if xa == 0.0 {
                continue;
            }
            for b in 0..d {
                m[a * d + b] += xa * row[b];
            }
        }
    }
    m
}

/// Subspace iteration on a symmetric PSD matrix `M ∈ R^{d × d}` to recover the
/// top-`r` eigenvectors as columns of `V ∈ R^{d × r}` (returned row-major as
/// `[d * r]`, i.e. `V[i*r + j]` is row `i` column `j`).
pub fn top_eigvecs(m: &[f32], d: usize, r: usize, iters: usize, seed: u64) -> Vec<f32> {
    assert!(r > 0 && r <= d);
    let mut rng = Lcg::new(seed);
    // Initialize V randomly, then orthonormalize.
    let mut v = vec![0.0_f32; d * r];
    for k in 0..d * r {
        v[k] = rng.next_f32() - 0.5;
    }
    gram_schmidt(&mut v, d, r);

    let mut tmp = vec![0.0_f32; d * r];
    for _ in 0..iters {
        // tmp = M @ V  ([d×d] @ [d×r]).
        for i in 0..d {
            for j in 0..r {
                let mut s = 0.0_f32;
                for k in 0..d {
                    s += m[i * d + k] * v[k * r + j];
                }
                tmp[i * r + j] = s;
            }
        }
        std::mem::swap(&mut v, &mut tmp);
        gram_schmidt(&mut v, d, r);
    }
    v
}

/// Modified Gram–Schmidt on the columns of a row-major `[d × r]` matrix.
pub fn gram_schmidt(v: &mut [f32], d: usize, r: usize) {
    for j in 0..r {
        // Normalize column j.
        let mut s = 0.0_f32;
        for i in 0..d {
            s += v[i * r + j] * v[i * r + j];
        }
        if s <= 0.0 {
            continue;
        }
        let inv = 1.0 / s.sqrt();
        for i in 0..d {
            v[i * r + j] *= inv;
        }
        // Subtract projection from later columns.
        for k in (j + 1)..r {
            let mut p = 0.0_f32;
            for i in 0..d {
                p += v[i * r + j] * v[i * r + k];
            }
            for i in 0..d {
                v[i * r + k] -= p * v[i * r + j];
            }
        }
    }
}

/// Multiply row-major `X ∈ R^{n × d}` by `V ∈ R^{d × r}`, output `[n × r]`.
pub fn matmul_xv(x: &[f32], n: usize, d: usize, v: &[f32], r: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n * d);
    debug_assert_eq!(v.len(), d * r);
    let mut a = vec![0.0_f32; n * r];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..d {
                s += row[k] * v[k * r + j];
            }
            a[i * r + j] = s;
        }
    }
    a
}

/// Project a single query `q ∈ R^d` into the rank-`r` subspace, output `[r]`.
pub fn project_q(q: &[f32], v: &[f32], d: usize, r: usize) -> Vec<f32> {
    debug_assert_eq!(q.len(), d);
    debug_assert_eq!(v.len(), d * r);
    let mut qr = vec![0.0_f32; r];
    for j in 0..r {
        let mut s = 0.0_f32;
        for i in 0..d {
            s += q[i] * v[i * r + j];
        }
        qr[j] = s;
    }
    qr
}

/// Estimate the fraction of cluster Frobenius energy captured by `V` —
/// `trace(V^T M V) / trace(M)`. Used as a diagnostic in the research doc.
pub fn captured_energy(m: &[f32], v: &[f32], d: usize, r: usize) -> f32 {
    let mut num = 0.0_f32;
    for j in 0..r {
        // (M V)[:,j]
        let mut mvj = vec![0.0_f32; d];
        for i in 0..d {
            let mut s = 0.0_f32;
            for k in 0..d {
                s += m[i * d + k] * v[k * r + j];
            }
            mvj[i] = s;
        }
        // (V[:,j])^T (M V)[:,j]
        for i in 0..d {
            num += v[i * r + j] * mvj[i];
        }
    }
    let mut tr = 0.0_f32;
    for i in 0..d {
        tr += m[i * d + i];
    }
    if tr > 0.0 {
        num / tr
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_schmidt_produces_orthonormal_columns() {
        // Use a well-conditioned random matrix (LCG-generated) — sin-sequence
        // columns are too collinear and collapse the third column to ~0,
        // producing numerical garbage after normalization.
        let d = 6;
        let r = 3;
        let mut rng = crate::Lcg::new(2026);
        let mut v: Vec<f32> = (0..d * r).map(|_| rng.next_f32() - 0.5).collect();
        gram_schmidt(&mut v, d, r);
        // Check V^T V ≈ I.
        for a in 0..r {
            for b in 0..r {
                let mut s = 0.0_f32;
                for i in 0..d {
                    s += v[i * r + a] * v[i * r + b];
                }
                let target = if a == b { 1.0 } else { 0.0 };
                assert!((s - target).abs() < 1e-4, "V^T V[{a},{b}] = {s}");
            }
        }
    }

    #[test]
    fn top_eigvecs_captures_energy_in_low_rank_data() {
        // Build an n×d matrix that lives in an r-dim subspace.
        let n = 40;
        let d = 8;
        let r_true = 3;
        let mut x = vec![0.0_f32; n * d];
        // Pick a random orthonormal basis U ∈ R^{d×r_true}.
        let mut rng = Lcg::new(11);
        let mut basis = vec![0.0_f32; d * r_true];
        for k in 0..d * r_true {
            basis[k] = rng.next_f32() - 0.5;
        }
        gram_schmidt(&mut basis, d, r_true);
        // x[i] = basis @ coeffs_i
        for i in 0..n {
            let coeffs: Vec<f32> = (0..r_true).map(|_| rng.next_f32() - 0.5).collect();
            for k in 0..d {
                let mut s = 0.0_f32;
                for j in 0..r_true {
                    s += basis[k * r_true + j] * coeffs[j];
                }
                x[i * d + k] = s;
            }
        }
        let m = gram(&x, n, d);
        let v = top_eigvecs(&m, d, r_true, 50, 3);
        let energy = captured_energy(&m, &v, d, r_true);
        assert!(
            energy > 0.98,
            "expected near-perfect energy capture for low-rank data, got {energy}"
        );
    }
}
