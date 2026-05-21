//! Lloyd k-means with k-means++ init. Operates on a single subspace.
//!
//! `weights[i]` is optional per-point weight (used by anisotropic-PQ to lift
//! parallel-component error). When `weights` is `None` we run unweighted Lloyd.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::metrics::sq_l2;

pub struct KmeansResult {
    pub centroids: Vec<Vec<f32>>, // length k, each length sub_dim
    pub assignments: Vec<u8>,     // length n
    pub inertia: f32,
}

pub fn kmeans_pp_init(
    points: &[Vec<f32>],
    k: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n = points.len();
    assert!(n >= k);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    chosen.push(first);

    let mut d2: Vec<f32> = points
        .iter()
        .map(|p| sq_l2(p, &points[first]))
        .collect();

    while chosen.len() < k {
        let total: f32 = d2.iter().sum();
        if total <= 0.0 {
            // duplicate points; just pick a random one
            let pick = rng.gen_range(0..n);
            chosen.push(pick);
            for j in 0..n {
                let nd = sq_l2(&points[j], &points[pick]);
                if nd < d2[j] {
                    d2[j] = nd;
                }
            }
            continue;
        }
        let mut threshold = rng.gen::<f32>() * total;
        let mut pick = n - 1;
        for (i, &w) in d2.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                pick = i;
                break;
            }
        }
        chosen.push(pick);
        for j in 0..n {
            let nd = sq_l2(&points[j], &points[pick]);
            if nd < d2[j] {
                d2[j] = nd;
            }
        }
    }

    chosen.iter().map(|&i| points[i].clone()).collect()
}

pub fn lloyd(
    points: &[Vec<f32>],
    weights: Option<&[f32]>,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> KmeansResult {
    let n = points.len();
    let sub_dim = points[0].len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeans_pp_init(points, k, &mut rng);
    let mut assignments = vec![0u8; n];
    let mut inertia = f32::INFINITY;

    for _iter in 0..max_iter {
        // assign
        let mut new_inertia = 0f32;
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = sq_l2(&points[i], c);
                if d < best_d {
                    best_d = d;
                    best = c_idx;
                }
            }
            assignments[i] = best as u8;
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            new_inertia += w * best_d;
        }

        // update
        let mut sums = vec![vec![0f32; sub_dim]; k];
        let mut counts = vec![0f32; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            counts[c] += w;
            for d in 0..sub_dim {
                sums[c][d] += w * points[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0.0 {
                let inv = 1.0 / counts[c];
                for d in 0..sub_dim {
                    centroids[c][d] = sums[c][d] * inv;
                }
            } else {
                // empty cluster: re-seed from a random point
                let pick = rng.gen_range(0..n);
                centroids[c] = points[pick].clone();
            }
        }

        // convergence check
        if (inertia - new_inertia).abs() < 1e-6 * inertia.max(1.0) {
            inertia = new_inertia;
            break;
        }
        inertia = new_inertia;
    }

    KmeansResult {
        centroids,
        assignments,
        inertia,
    }
}

/// Convenience: choose `k` distinct random points (fallback when n is tiny).
pub fn random_init(points: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..points.len()).collect();
    idx.shuffle(&mut rng);
    idx.into_iter().take(k).map(|i| points[i].clone()).collect()
}
