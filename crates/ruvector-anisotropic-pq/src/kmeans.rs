//! k-means with pluggable per-example loss, used for per-subspace codebook
//! training in PQ. Anisotropic loss weights the residual component parallel
//! to the parent (full-vector) direction — the key ScaNN insight.
//!
//! The training loop is deterministic given `seed`. Assignment and centroid
//! update are parallelised with rayon. Loss stays under 500 lines.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Per-subvector weight configuration used during codebook training.
/// `w_parallel` and `w_perp` weight the two residual components.
#[derive(Copy, Clone, Debug)]
pub struct SubLossWeights {
    pub w_parallel: f32,
    pub w_perp: f32,
}

impl SubLossWeights {
    #[inline]
    pub fn reconstruction() -> Self {
        Self { w_parallel: 1.0, w_perp: 1.0 }
    }
}

/// Train K centroids for one subspace.
///
/// * `sub` — flat n*sub_d subvectors for this subspace.
/// * `parent_dir` — flat n*sub_d unit vectors representing the parent
///   direction restricted to this subspace (used only when weights aren't
///   isotropic). Pass zeros for pure reconstruction training.
/// * `weights` — per-example (w_parallel, w_perp).
///
/// Returns `k * sub_d` centroids row-major.
pub fn train_subspace(
    sub: &[f32],
    parent_dir: &[f32],
    weights: &[SubLossWeights],
    sub_d: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> Vec<f32> {
    let n = sub.len() / sub_d;
    assert_eq!(parent_dir.len(), n * sub_d);
    assert_eq!(weights.len(), n);
    assert!(n >= k, "n={n} k={k}");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centroids = kmeanspp_init(sub, sub_d, k, &mut rng);
    let mut assign = vec![0u32; n];

    for _ in 0..iters {
        // Assignment step (parallel).
        assign.par_iter_mut().enumerate().for_each(|(i, a)| {
            let x = &sub[i * sub_d..(i + 1) * sub_d];
            let d = &parent_dir[i * sub_d..(i + 1) * sub_d];
            let w = weights[i];
            let mut best = f32::MAX;
            let mut best_c = 0u32;
            for c in 0..k {
                let cent = &centroids[c * sub_d..(c + 1) * sub_d];
                let loss = weighted_loss(x, cent, d, w);
                if loss < best {
                    best = loss;
                    best_c = c as u32;
                }
            }
            *a = best_c;
        });

        // Update step: weighted mean minimizes the anisotropic quadratic per
        // subspace only in the isotropic special case. For anisotropic loss we
        // apply a whitening-style correction on the parent-direction component.
        centroids = update_centroids(sub, parent_dir, weights, &assign, sub_d, k);
    }
    centroids
}

fn kmeanspp_init(sub: &[f32], sub_d: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let n = sub.len() / sub_d;
    let mut centroids = Vec::with_capacity(k * sub_d);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&sub[first * sub_d..(first + 1) * sub_d]);
    let mut dists = vec![f32::MAX; n];
    for _ in 1..k {
        // Update squared distance to nearest centroid.
        let last = centroids.len() / sub_d - 1;
        let last_c = &centroids[last * sub_d..(last + 1) * sub_d];
        for i in 0..n {
            let x = &sub[i * sub_d..(i + 1) * sub_d];
            let d = l2_sq(x, last_c);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        // Sample proportional to squared distance.
        let total: f32 = dists.iter().sum();
        let mut r: f32 = rng.gen::<f32>() * total.max(1e-12);
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            r -= d;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.extend_from_slice(&sub[chosen * sub_d..(chosen + 1) * sub_d]);
    }
    centroids
}

fn update_centroids(
    sub: &[f32],
    parent_dir: &[f32],
    weights: &[SubLossWeights],
    assign: &[u32],
    sub_d: usize,
    k: usize,
) -> Vec<f32> {
    // Anisotropic MSE optimum per cluster requires solving A_c · c = b_c,
    // where
    //   A_c = M · I + (η - 1) · Σ d_i d_iᵀ
    //   b_c = Σ x_i + (η - 1) · Σ (x_i · d_i) d_i
    // Standard k-means is recovered when η = 1.
    let n = sub.len() / sub_d;
    let anisotropic = weights.iter().any(|w| (w.w_parallel - w.w_perp).abs() > 1e-6);

    // Per-cluster accumulators.
    let mat_sz = sub_d * sub_d;
    let mut a_mats: Vec<Vec<f64>> = (0..k).map(|_| vec![0f64; mat_sz]).collect();
    let mut b_vecs: Vec<Vec<f64>> = (0..k).map(|_| vec![0f64; sub_d]).collect();
    let mut counts = vec![0f64; k];

    for i in 0..n {
        let c = assign[i] as usize;
        let x = &sub[i * sub_d..(i + 1) * sub_d];
        let w = weights[i];
        counts[c] += 1.0;
        let a = &mut a_mats[c];
        let b = &mut b_vecs[c];
        for j in 0..sub_d {
            b[j] += x[j] as f64;
        }
        if anisotropic {
            let d = &parent_dir[i * sub_d..(i + 1) * sub_d];
            let eta_m1 = (w.w_parallel - w.w_perp) as f64;
            let xd: f64 = (0..sub_d).map(|j| (x[j] * d[j]) as f64).sum();
            for j in 0..sub_d {
                b[j] += eta_m1 * xd * d[j] as f64;
                for jj in 0..sub_d {
                    a[j * sub_d + jj] += eta_m1 * (d[j] * d[jj]) as f64;
                }
            }
        }
    }

    let mut cent = vec![0f32; k * sub_d];
    for c in 0..k {
        let m_cnt = counts[c].max(1e-12);
        // Add M·I to A.
        for j in 0..sub_d {
            a_mats[c][j * sub_d + j] += m_cnt;
        }
        // Solve A c = b via Cholesky (A is SPD when eta>=1).
        let sol = if anisotropic {
            solve_spd(&a_mats[c], &b_vecs[c], sub_d).unwrap_or_else(|| {
                // Fallback: mean.
                let mut m = vec![0f64; sub_d];
                for j in 0..sub_d { m[j] = b_vecs[c][j] / m_cnt; }
                m
            })
        } else {
            let mut m = vec![0f64; sub_d];
            for j in 0..sub_d { m[j] = b_vecs[c][j] / m_cnt; }
            m
        };
        for j in 0..sub_d {
            cent[c * sub_d + j] = sol[j] as f32;
        }
    }
    cent
}

/// Solve SPD system A x = b via Cholesky. Returns None on failure.
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0f64; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for kk in 0..j {
            s -= l[j * n + kk] * l[j * n + kk];
        }
        if s <= 0.0 { return None; }
        l[j * n + j] = s.sqrt();
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for kk in 0..j {
                s -= l[i * n + kk] * l[j * n + kk];
            }
            l[i * n + j] = s / l[j * n + j];
        }
    }
    // Forward sub: L y = b
    let mut y = vec![0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for kk in 0..i {
            s -= l[i * n + kk] * y[kk];
        }
        y[i] = s / l[i * n + i];
    }
    // Back sub: L^T x = y
    let mut x = vec![0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for kk in (i + 1)..n {
            s -= l[kk * n + i] * x[kk];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Weighted loss between subvector x and candidate centroid c, given the
/// parent-direction restriction `d`. When `w.w_parallel == w.w_perp` this
/// reduces to squared L2. Otherwise the residual is decomposed into
/// parallel/perp components and each is weighted separately.
#[inline]
pub fn weighted_loss(x: &[f32], c: &[f32], d: &[f32], w: SubLossWeights) -> f32 {
    if (w.w_parallel - w.w_perp).abs() < 1e-6 {
        return l2_sq(x, c) * w.w_perp;
    }
    // r = x - c
    // r_parallel = (r · d) * d  (d expected unit-norm on this subspace, or
    // approximately so; small residual on that assumption is acceptable).
    let mut rd = 0f32;
    for i in 0..x.len() {
        rd += (x[i] - c[i]) * d[i];
    }
    let mut par_sq = 0f32;
    let mut perp_sq = 0f32;
    for i in 0..x.len() {
        let r = x[i] - c[i];
        let par = rd * d[i];
        let per = r - par;
        par_sq += par * par;
        perp_sq += per * per;
    }
    par_sq * w.w_parallel + perp_sq * w.w_perp
}
