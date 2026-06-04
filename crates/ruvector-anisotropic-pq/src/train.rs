//! Training: plain L2 k-means PQ and anisotropic-weighted k-means PQ.
//!
//! The anisotropic update solves a small linear system per centroid per
//! subspace per iteration. With η = 1 the system reduces to the L2 normal
//! equations and the closed-form mean — the two trainers stay numerically
//! consistent at the η = 1 boundary.

use super::AnisotropicPq;
use rand::Rng;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug)]
pub struct TrainOpts {
    pub iters: usize,
    pub k: usize,
    pub seed: u64,
}

impl Default for TrainOpts {
    fn default() -> Self {
        Self { iters: 12, k: 256, seed: 42 }
    }
}

/// Train a plain L2 PQ codebook (equivalent to η = 1).
pub fn train_l2_pq(data: &[f32], n: usize, dim: usize, m: usize, k: usize, opts: TrainOpts) -> AnisotropicPq {
    train_anisotropic_pq(data, n, dim, m, k, 1.0, opts)
}

/// Train an anisotropic PQ codebook with anisotropy ratio `eta` ≥ 1.
///
/// Block-diagonal approximation of ScaNN's anisotropic loss for unit-norm data:
///   L = (η-1)(e·x̂)² + ||e||²  with x̂ = x/||x||,
/// drops cross-subspace terms to obtain a per-subspace weight matrix
///   W_i,s = (η - 1) (x_i,s / ||x_i||) (x_i,s / ||x_i||)^T + I.
/// For unit-norm x this simplifies to W_i,s = (η - 1) x_i,s x_i,s^T + I,
/// where x_i,s is the subvector slice — NOT the locally-normalised one.
/// The effective parallel-amplification in subspace s is therefore
/// 1 + (η - 1)||x_s||² — subspaces with more mass get bent harder toward the
/// data manifold, matching the global loss they contribute to.
///
/// - Assignment: argmin_c (x_i,s - μ_c,s)^T W_i,s (x_i,s - μ_c,s)
/// - Update: per cluster solve (Σ_i W_i,s) μ_c,s = Σ_i W_i,s x_i,s
///
/// At η = 1, W = I and the update collapses to the arithmetic mean.
pub fn train_anisotropic_pq(
    data: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    k: usize,
    eta: f32,
    opts: TrainOpts,
) -> AnisotropicPq {
    assert!(eta >= 1.0, "η must be ≥ 1 (η=1 is L2 PQ)");
    assert_eq!(dim % m, 0, "dim must be divisible by m");
    let d_sub = dim / m;
    assert!(k >= 2 && k <= 256);

    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(opts.seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(m);

    for s in 0..m {
        // Initialise: k random distinct training points' s-th subvector.
        let mut init: Vec<f32> = Vec::with_capacity(k * d_sub);
        let mut picked = std::collections::HashSet::new();
        while picked.len() < k {
            let idx = rng.gen_range(0..n);
            if picked.insert(idx) {
                let off = idx * dim + s * d_sub;
                init.extend_from_slice(&data[off..off + d_sub]);
            }
        }

        let cs = train_subspace(data, n, dim, s, d_sub, init, k, eta, opts.iters);
        centroids.push(cs);
    }

    AnisotropicPq { dim, m, d_sub, k, eta, centroids }
}

fn train_subspace(
    data: &[f32],
    n: usize,
    dim: usize,
    s: usize,
    d_sub: usize,
    mut centroids: Vec<f32>,
    k: usize,
    eta: f32,
    iters: usize,
) -> Vec<f32> {
    // Pre-extract subspace slice. Also pre-divide by global vector norm so that
    // `uhat_i,s` := x_i,s / ||x_i||, the slice of the GLOBAL unit direction
    // restricted to subspace s.  For unit-norm input this equals x_i,s.
    let mut sub = vec![0f32; n * d_sub];
    let mut uhat = vec![0f32; n * d_sub];
    for i in 0..n {
        // global norm of x_i across all dims.
        let row = &data[i * dim..(i + 1) * dim];
        let mut g_norm = 0f32;
        for v in row {
            g_norm += *v * *v;
        }
        let g_norm = g_norm.sqrt().max(1e-9);
        let off = i * dim + s * d_sub;
        let src = &data[off..off + d_sub];
        sub[i * d_sub..(i + 1) * d_sub].copy_from_slice(src);
        for j in 0..d_sub {
            uhat[i * d_sub + j] = src[j] / g_norm;
        }
    }

    let mut assign = vec![0u16; n];

    for _it in 0..iters {
        // --- E-step: assign each x_i to argmin anisotropic distance.
        for i in 0..n {
            let xi = &sub[i * d_sub..(i + 1) * d_sub];
            let ui = &uhat[i * d_sub..(i + 1) * d_sub];
            let mut best_c = 0u16;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cv = &centroids[c * d_sub..(c + 1) * d_sub];
                // d² = (η-1)((x-c)·û)² + ||x-c||²
                let mut diff_dot_u = 0f32;
                let mut sq = 0f32;
                for j in 0..d_sub {
                    let d = xi[j] - cv[j];
                    diff_dot_u += d * ui[j];
                    sq += d * d;
                }
                let val = (eta - 1.0) * diff_dot_u * diff_dot_u + sq;
                if val < best_d {
                    best_d = val;
                    best_c = c as u16;
                }
            }
            assign[i] = best_c;
        }

        // --- M-step: per cluster, solve (Σ W_i) c = Σ W_i x_i.
        // W_i = (η-1) û û^T + I → ΣW = (η-1) Σ ûûᵀ + |C| I.
        // For small d_sub (typ. ≤ 16) we solve an explicit Gauss elimination.
        let cap_m = d_sub * d_sub;
        let mut a = vec![0f32; cap_m];
        let mut rhs = vec![0f32; d_sub];

        // Buckets: per cluster sums.
        // For memory, redo per cluster in a sweep over n.
        // Build accumulators flat: sum_uu[k * d² + ...], sum_x[k*d_sub], count[k].
        let mut sum_uu = vec![0f32; k * cap_m];
        let mut sum_wx = vec![0f32; k * d_sub];
        let mut count = vec![0u32; k];

        for i in 0..n {
            let c = assign[i] as usize;
            count[c] += 1;
            let xi = &sub[i * d_sub..(i + 1) * d_sub];
            let ui = &uhat[i * d_sub..(i + 1) * d_sub];
            let dot_u_x = {
                let mut acc = 0f32;
                for j in 0..d_sub {
                    acc += ui[j] * xi[j];
                }
                acc
            };
            // Σ W x = Σ ((η-1) û (ûᵀx) + x) = (η-1) (ûᵀx) û + x
            let kk = eta - 1.0;
            let off_x = c * d_sub;
            for j in 0..d_sub {
                sum_wx[off_x + j] += kk * dot_u_x * ui[j] + xi[j];
            }
            // Σ ûûᵀ accumulator.
            let off_uu = c * cap_m;
            for a_i in 0..d_sub {
                let u_a = ui[a_i];
                for a_j in 0..d_sub {
                    sum_uu[off_uu + a_i * d_sub + a_j] += u_a * ui[a_j];
                }
            }
        }

        for c in 0..k {
            if count[c] == 0 {
                continue; // keep prior centroid
            }
            // Build A = (η-1) Σ ûûᵀ + |C| I
            let n_c = count[c] as f32;
            let off_uu = c * cap_m;
            for r in 0..d_sub {
                for col in 0..d_sub {
                    a[r * d_sub + col] = (eta - 1.0) * sum_uu[off_uu + r * d_sub + col];
                }
                a[r * d_sub + r] += n_c;
            }
            for j in 0..d_sub {
                rhs[j] = sum_wx[c * d_sub + j];
            }
            // Solve A x = rhs in place (Gaussian elimination with partial pivoting).
            if solve_in_place(&mut a, &mut rhs, d_sub).is_some() {
                let off_c = c * d_sub;
                for j in 0..d_sub {
                    centroids[off_c + j] = rhs[j];
                }
            }
            // If singular (degenerate), keep existing centroid.
        }
    }

    centroids
}

/// Gaussian elimination with partial pivoting. Solves A x = b, writes x into b.
/// Returns None if singular.
fn solve_in_place(a: &mut [f32], b: &mut [f32], n: usize) -> Option<()> {
    for k in 0..n {
        // Pivot.
        let mut max_v = a[k * n + k].abs();
        let mut pivot = k;
        for r in (k + 1)..n {
            let v = a[r * n + k].abs();
            if v > max_v {
                max_v = v;
                pivot = r;
            }
        }
        if max_v < 1e-12 {
            return None;
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            b.swap(k, pivot);
        }
        // Eliminate.
        let pinv = 1.0 / a[k * n + k];
        for r in (k + 1)..n {
            let factor = a[r * n + k] * pinv;
            if factor == 0.0 {
                continue;
            }
            for c in k..n {
                a[r * n + c] -= factor * a[k * n + c];
            }
            b[r] -= factor * b[k];
        }
    }
    // Back-substitute.
    for k in (0..n).rev() {
        let mut s = b[k];
        for c in (k + 1)..n {
            s -= a[k * n + c] * b[c];
        }
        b[k] = s / a[k * n + k];
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_identity() {
        // A = I, b = [1,2,3] → x = b.
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = vec![1.0, 2.0, 3.0];
        solve_in_place(&mut a, &mut b, 3).expect("identity solve");
        assert_eq!(b, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn solve_diag() {
        let mut a = vec![2.0, 0.0, 0.0, 4.0];
        let mut b = vec![6.0, 8.0];
        solve_in_place(&mut a, &mut b, 2).expect("diag solve");
        assert!((b[0] - 3.0).abs() < 1e-6);
        assert!((b[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn eta_one_converges() {
        // With η = 1 (vanilla L2 k-means) the trainer must converge: training
        // for N+5 iterations from the same seed must produce centroids
        // within ε of training for N iterations (Lloyd's is monotone and
        // converges in finitely many steps for finite data).
        use crate::synthetic_unit_dataset;
        let n = 1_000;
        let dim = 8;
        let data = synthetic_unit_dataset(n, dim, 11);
        let a = train_l2_pq(&data, n, dim, 2, 16, TrainOpts { iters: 80, k: 16, seed: 3 });
        let b = train_l2_pq(&data, n, dim, 2, 16, TrainOpts { iters: 90, k: 16, seed: 3 });
        let mut max_d = 0f32;
        for s in 0..a.centroids.len() {
            for j in 0..a.centroids[s].len() {
                let d = (a.centroids[s][j] - b.centroids[s][j]).abs();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        assert!(max_d < 5e-2, "L2 PQ should be close to convergence; max Δ between 80 and 90 iters = {}", max_d);
    }
}
