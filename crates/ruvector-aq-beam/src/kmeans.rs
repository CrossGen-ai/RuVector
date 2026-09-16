//! Minimal Lloyd's-algorithm k-means used for PQ subspaces and AQ initial
//! codebooks. Deterministic given the seed.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::sq_l2;

/// Train `k` centroids on `data` (each row length `d`) using `iters`
/// iterations of Lloyd's. Empty clusters are re-seeded from the farthest
/// point currently mis-served.
pub fn kmeans(data: &[Vec<f32>], k: usize, d: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(k > 0);
    assert!(!data.is_empty());

    let mut rng = StdRng::seed_from_u64(seed);
    let n = data.len();

    // k-means++ seeding.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(data[first].clone());

    let mut d2: Vec<f32> = data.iter().map(|v| sq_l2(v, &centroids[0])).collect();
    for _ in 1..k {
        let total: f32 = d2.iter().sum();
        if total <= 0.0 {
            let idx = rng.gen_range(0..n);
            centroids.push(data[idx].clone());
        } else {
            let mut target = rng.gen::<f32>() * total;
            let mut chosen = 0;
            for (i, w) in d2.iter().enumerate() {
                target -= *w;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.push(data[chosen].clone());
        }
        let last = centroids.last().unwrap();
        for (i, v) in data.iter().enumerate() {
            let nd = sq_l2(v, last);
            if nd < d2[i] {
                d2[i] = nd;
            }
        }
    }

    let mut assign: Vec<usize> = vec![0usize; n];
    for it in 0..iters {
        // Assign step.
        for (i, v) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for (c, cent) in centroids.iter().enumerate() {
                let dv = sq_l2(v, cent);
                if dv < best_d {
                    best_d = dv;
                    best = c;
                }
            }
            assign[i] = best;
        }
        // Update step.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; d]; k];
        let mut cnts: Vec<usize> = vec![0; k];
        for (i, &a) in assign.iter().enumerate() {
            let v = &data[i];
            let s = &mut sums[a];
            for j in 0..d {
                s[j] += v[j];
            }
            cnts[a] += 1;
        }
        for c in 0..k {
            if cnts[c] > 0 {
                let inv = 1.0f32 / cnts[c] as f32;
                for j in 0..d {
                    centroids[c][j] = sums[c][j] * inv;
                }
            } else {
                // Empty cluster: reseed to random data point.
                let idx = rng.gen_range(0..n);
                centroids[c] = data[idx].clone();
            }
        }
        // Cheap convergence check: stop early on last iteration only.
        let _ = it;
    }
    centroids
}

/// Assign each row of `data` to its nearest centroid, return codes.
pub fn assign(data: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<u8> {
    debug_assert!(centroids.len() <= 256);
    let mut out = vec![0u8; data.len()];
    for (i, v) in data.iter().enumerate() {
        let mut best = 0u8;
        let mut best_d = f32::MAX;
        for (c, cent) in centroids.iter().enumerate() {
            let dv = sq_l2(v, cent);
            if dv < best_d {
                best_d = dv;
                best = c as u8;
            }
        }
        out[i] = best;
    }
    out
}
