//! Anisotropic k-means: minimises the score-aware loss
//! `η · ||e_∥||² + ||e_⊥||²` per Guo et al. (ScaNN, ICML 2020, Eq. 12).
//!
//! Here `e = x − c`, `x̂ = x / ||x||`, `e_∥ = (e·x̂) x̂`, `e_⊥ = e − e_∥`.
//!
//! Per assignment step we score points with the anisotropic loss; per
//! update step we solve the (small) `d_sub × d_sub` linear system
//! `[N I + (η−1) A] c = sum(x) + (η−1) Σ ||x_i|| x̂_i` where
//! `A = Σ x̂_i x̂_i^T`. `d_sub` is small (`dim / M`, e.g. 8), so Gaussian
//! elimination is fine.

use crate::kmeans::{kmeans_pp_wrap, sq_dist};
use crate::AvqError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Train `k` anisotropic centroids on a slice of points. `eta` is the
/// parallel-vs-orthogonal weight ratio; `eta = 1.0` reduces to plain
/// MSE k-means.
pub fn train_avq(
    data: &[Vec<f32>],
    k: usize,
    iters: usize,
    eta: f32,
    seed: u64,
) -> Result<Vec<Vec<f32>>, AvqError> {
    let n = data.len();
    if n == 0 {
        return Err(AvqError::EmptyTraining);
    }
    if n < k {
        return Err(AvqError::NotEnoughPoints { k, n });
    }
    let dim = data[0].len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeans_pp_wrap(data, k, &mut rng);

    // Cache unit-vectors and norms for anisotropic math.
    let (hats, norms) = precompute_directions(data);

    for _ in 0..iters {
        // Assignment: minimise anisotropic loss per point.
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, x) in data.iter().enumerate() {
            let a = assign_anisotropic(x, &hats[i], &centroids, eta);
            buckets[a].push(i);
        }
        // Update: solve linear system per cluster.
        for c in 0..k {
            if buckets[c].is_empty() {
                centroids[c] = data[rng.gen_range(0..n)].clone();
                continue;
            }
            centroids[c] = solve_anisotropic(&buckets[c], data, &hats, &norms, eta, dim);
        }
    }
    Ok(centroids)
}

pub fn precompute_directions(data: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut hats: Vec<Vec<f32>> = Vec::with_capacity(data.len());
    let mut norms: Vec<f32> = Vec::with_capacity(data.len());
    for x in data {
        let n2: f32 = x.iter().map(|v| v * v).sum();
        let n = n2.sqrt();
        norms.push(n);
        if n > 0.0 {
            hats.push(x.iter().map(|v| v / n).collect());
        } else {
            hats.push(vec![0.0; x.len()]);
        }
    }
    (hats, norms)
}

/// Anisotropic loss `η ||e_∥||² + ||e_⊥||²`. If `x̂` is zero (norm 0),
/// the loss degenerates to plain squared distance.
pub fn anisotropic_loss(x: &[f32], x_hat: &[f32], c: &[f32], eta: f32) -> f32 {
    let mut e = 0.0f32;      // ||e||²
    let mut par = 0.0f32;    // e · x̂
    for i in 0..x.len() {
        let d = x[i] - c[i];
        e += d * d;
        par += d * x_hat[i];
    }
    let par_sq = par * par; // ||e_∥||² since x̂ is unit
    let orth_sq = (e - par_sq).max(0.0);
    eta * par_sq + orth_sq
}

fn assign_anisotropic(x: &[f32], x_hat: &[f32], centroids: &[Vec<f32>], eta: f32) -> usize {
    let mut best = 0;
    let mut best_l = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let l = if x_hat.iter().all(|v| *v == 0.0) {
            sq_dist(x, c)
        } else {
            anisotropic_loss(x, x_hat, c, eta)
        };
        if l < best_l {
            best_l = l;
            best = i;
        }
    }
    best
}

/// Solve `[N I + (η−1) A] c = Σ x_i + (η−1) Σ ||x_i|| x̂_i` where the
/// sum is over the cluster and `A = Σ x̂_i x̂_i^T`.
fn solve_anisotropic(
    idxs: &[usize],
    data: &[Vec<f32>],
    hats: &[Vec<f32>],
    norms: &[f32],
    eta: f32,
    dim: usize,
) -> Vec<f32> {
    let n = idxs.len() as f32;
    let alpha = eta - 1.0;

    let mut a = vec![vec![0.0f32; dim]; dim]; // LHS matrix
    let mut b = vec![0.0f32; dim];             // RHS vector

    // Diagonal N·I term.
    for i in 0..dim {
        a[i][i] = n;
    }
    // (η − 1) · A term.
    if alpha.abs() > 1e-8 {
        for &idx in idxs {
            let h = &hats[idx];
            for i in 0..dim {
                for j in 0..dim {
                    a[i][j] += alpha * h[i] * h[j];
                }
            }
        }
    }
    // RHS: Σ x + (η−1) Σ ||x_i|| x̂_i.
    for &idx in idxs {
        let x = &data[idx];
        for i in 0..dim {
            b[i] += x[i];
        }
        if alpha.abs() > 1e-8 {
            let nrm = norms[idx];
            let h = &hats[idx];
            for i in 0..dim {
                b[i] += alpha * nrm * h[i];
            }
        }
    }
    gauss_solve(&mut a, &mut b)
}

/// Naive partial-pivot Gaussian elimination. `dim` is small (≤ 32),
/// so this is negligible.
fn gauss_solve(a: &mut [Vec<f32>], b: &mut [f32]) -> Vec<f32> {
    let n = b.len();
    for i in 0..n {
        // Pivot: swap in the largest row below.
        let mut piv = i;
        let mut best = a[i][i].abs();
        for r in (i + 1)..n {
            if a[r][i].abs() > best {
                best = a[r][i].abs();
                piv = r;
            }
        }
        if piv != i {
            a.swap(i, piv);
            b.swap(i, piv);
        }
        let d = a[i][i];
        if d.abs() < 1e-12 {
            // Singular — fall back to a zero contribution (very rare
            // with anisotropic regularisation).
            continue;
        }
        for r in (i + 1)..n {
            let factor = a[r][i] / d;
            if factor == 0.0 {
                continue;
            }
            for c in i..n {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        let d = a[i][i];
        x[i] = if d.abs() < 1e-12 { 0.0 } else { sum / d };
    }
    x
}
