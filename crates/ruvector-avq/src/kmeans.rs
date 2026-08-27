//! Plain Lloyd k-means (MSE) with deterministic k-means++ init.

use crate::{dot, AvqError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn train_mse(
    data: &[Vec<f32>],
    k: usize,
    iters: usize,
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
    let mut centroids = kmeans_pp(data, k, &mut rng);

    for _ in 0..iters {
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for x in data {
            let a = assign(x, &centroids);
            for (s, v) in sums[a].iter_mut().zip(x) {
                *s += *v;
            }
            counts[a] += 1;
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty cluster with a random point.
                let idx = rng.gen_range(0..n);
                centroids[c] = data[idx].clone();
            } else {
                let inv = 1.0 / counts[c] as f32;
                for v in &mut sums[c] {
                    *v *= inv;
                }
                centroids[c] = sums[c].clone();
            }
        }
    }
    Ok(centroids)
}

pub fn kmeans_pp_wrap(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    kmeans_pp(data, k, rng)
}

fn kmeans_pp(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(data[first].clone());

    let mut dists = vec![f32::INFINITY; n];
    while centroids.len() < k {
        let last = centroids.last().unwrap();
        let mut total = 0.0;
        for (i, x) in data.iter().enumerate() {
            let d = sq_dist(x, last);
            if d < dists[i] {
                dists[i] = d;
            }
            total += dists[i];
        }
        if total <= 0.0 {
            centroids.push(data[rng.gen_range(0..n)].clone());
            continue;
        }
        let mut target = rng.gen::<f32>() * total;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            target -= d;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }
    centroids
}

#[inline]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        s += d * d;
    }
    s
}

#[inline]
pub fn assign(x: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist(x, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Utility exposed so higher-level code can compute a plain inner
/// product against a set of centroids (used in ADC LUT construction).
#[inline]
pub fn lut_ip(query_sub: &[f32], centroids: &[Vec<f32>]) -> Vec<f32> {
    centroids.iter().map(|c| dot(query_sub, c)).collect()
}
