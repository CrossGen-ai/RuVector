//! Deterministic k-means variants used by the PQ codebook trainers.
//!
//! * [`lloyd`] — standard Lloyd's k-means with k-means++ init.
//! * [`anisotropic`] — score-aware weighted k-means (Guo et al., 2020).
//!
//! Both return `k` centroids of the same dimension as the training subvectors.
//! Deterministic given the seed: k-means++ init uses the same `StdRng`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::sq_l2;

/// Standard k-means++ init + Lloyd's iterations.
///
/// If `k` exceeds `data.len()` the centroid table is padded by repeating the
/// last observed vector — callers should have already validated `k`, this is
/// only a defensive fallback so training never panics on toy corpora.
pub fn lloyd(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kpp_init(data, k, &mut rng);
    let ds = centroids[0].len();
    for _ in 0..iters {
        let mut sums: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; ds]).collect();
        let mut cnts = vec![0usize; k];
        for x in data {
            let a = nearest(x, &centroids);
            for j in 0..ds {
                sums[a][j] += x[j];
            }
            cnts[a] += 1;
        }
        for i in 0..k {
            if cnts[i] > 0 {
                let inv = 1.0 / cnts[i] as f32;
                for j in 0..ds {
                    centroids[i][j] = sums[i][j] * inv;
                }
            }
        }
    }
    centroids
}

/// Anisotropic weighted k-means.
///
/// For each subvector `x` with direction `d = x / ||x||` (subspace-restricted),
/// we split residuals `r = x - c` into parallel/orthogonal components w.r.t.
/// `d`:
///
/// * `alpha = r · d`
/// * `r_par = alpha * d`,  `r_orth = r - r_par`
/// * loss   = `eta * ||r_par||^2 + ||r_orth||^2`
///
/// Assignment: pick the centroid minimising this loss.
/// Update: per Guo et al. §3.3, the h-weighted centroid update reduces to a
/// simple weighted average when we treat the eta factor as a per-sample
/// weight along `d`. We use the closed form:
///
/// ```text
/// c_i = (Σ_{x∈S_i} W_x x) / (Σ_{x∈S_i} W_x)
/// ```
///
/// where `W_x = 1 + (eta - 1) * (d d^T)` acts on `x` in matrix form. In code
/// this is `w * (α d) + (x - α d)` accumulated per-coordinate.
pub fn anisotropic(
    data: &[Vec<f32>],
    dirs: &[Vec<f32>],
    k: usize,
    iters: usize,
    eta: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    assert_eq!(data.len(), dirs.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kpp_init(data, k, &mut rng);
    let ds = centroids[0].len();

    for _ in 0..iters {
        // Per-cluster small linear systems:
        //   A c = b
        // where
        //   A = Σ_{x∈S} (I + (η-1) d_x d_x^T)     (ds × ds)
        //   b = Σ_{x∈S} (I + (η-1) d_x d_x^T) x   (ds)
        // We solve each with Gaussian elimination (ds is small, typically 4-16).
        let mut a_mats: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; ds * ds]).collect();
        let mut b_vecs: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; ds]).collect();
        let mut counts = vec![0usize; k];

        for (x, d) in data.iter().zip(dirs.iter()) {
            let assign = nearest_aniso(x, d, &centroids, eta);
            counts[assign] += 1;
            let am = &mut a_mats[assign];
            let bv = &mut b_vecs[assign];
            // A += I + (η-1) d d^T
            for i in 0..ds {
                am[i * ds + i] += 1.0;
                for j in 0..ds {
                    am[i * ds + j] += (eta - 1.0) * d[i] * d[j];
                }
            }
            // b += x + (η-1) (d·x) d
            let dx: f32 = (0..ds).map(|j| d[j] * x[j]).sum();
            for j in 0..ds {
                bv[j] += x[j] + (eta - 1.0) * dx * d[j];
            }
        }

        for i in 0..k {
            if counts[i] == 0 {
                continue;
            }
            if let Some(c) = solve_small(&a_mats[i], &b_vecs[i], ds) {
                centroids[i] = c;
            }
        }
    }
    centroids
}

fn kpp_init(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let ds = data[0].len();
    let mut chosen: Vec<Vec<f32>> = Vec::with_capacity(k);
    // First centroid: uniform sample.
    chosen.push(data[rng.gen_range(0..data.len())].clone());
    let mut dists: Vec<f32> = data
        .iter()
        .map(|x| sq_l2(x, &chosen[0]))
        .collect();
    while chosen.len() < k {
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            // All points identical to existing centroids — pad with copies.
            chosen.push(chosen[0].clone());
            continue;
        }
        let mut r: f32 = rng.gen_range(0.0..total);
        let mut idx = 0usize;
        for (i, d) in dists.iter().enumerate() {
            r -= *d;
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        chosen.push(data[idx].clone());
        // Update dist^2 to nearest chosen centroid.
        for (i, x) in data.iter().enumerate() {
            let nd = sq_l2(x, &chosen[chosen.len() - 1]);
            if nd < dists[i] {
                dists[i] = nd;
            }
        }
    }
    // Defensive pad (shouldn't trigger with k≤data.len()).
    while chosen.len() < k {
        chosen.push(vec![0.0f32; ds]);
    }
    chosen
}

fn nearest(x: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_l2(x, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Gaussian elimination for small symmetric-positive-definite systems.
/// Returns `Some(x)` such that `A x = b`, or `None` if singular.
fn solve_small(a_row_major: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut m = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a_row_major[i * n + j];
        }
        m[i * (n + 1) + n] = b[i];
    }
    for i in 0..n {
        // Partial pivot.
        let mut max_row = i;
        let mut max_val = m[i * (n + 1) + i].abs();
        for r in (i + 1)..n {
            let v = m[r * (n + 1) + i].abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-9 {
            return None;
        }
        if max_row != i {
            for c in 0..=n {
                m.swap(i * (n + 1) + c, max_row * (n + 1) + c);
            }
        }
        let pivot = m[i * (n + 1) + i];
        for r in (i + 1)..n {
            let f = m[r * (n + 1) + i] / pivot;
            for c in i..=n {
                m[r * (n + 1) + c] -= f * m[i * (n + 1) + c];
            }
        }
    }
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut s = m[i * (n + 1) + n];
        for c in (i + 1)..n {
            s -= m[i * (n + 1) + c] * x[c];
        }
        x[i] = s / m[i * (n + 1) + i];
    }
    Some(x)
}

fn nearest_aniso(x: &[f32], dir: &[f32], centroids: &[Vec<f32>], eta: f32) -> usize {
    let mut best = 0;
    let mut best_l = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        // r = x - c
        // alpha = r · dir
        // ||r_par||^2 = alpha^2 (dir is unit)
        // ||r_orth||^2 = ||r||^2 - alpha^2
        let mut alpha = 0.0f32;
        let mut r2 = 0.0f32;
        for j in 0..x.len() {
            let r = x[j] - c[j];
            alpha += r * dir[j];
            r2 += r * r;
        }
        let par2 = alpha * alpha;
        let orth2 = (r2 - par2).max(0.0);
        let loss = eta * par2 + orth2;
        if loss < best_l {
            best_l = loss;
            best = i;
        }
    }
    best
}
