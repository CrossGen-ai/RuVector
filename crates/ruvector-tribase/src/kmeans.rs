//! Minimal Lloyd k-means with k-means++ init. CPU-only, deterministic via seeded RNG.

use crate::dist::sq_l2;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<u32>,
}

pub fn fit(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> KMeansResult {
    assert!(!data.is_empty());
    assert!(k > 0 && k <= data.len());
    let d = data[0].len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // k-means++ init
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..data.len());
    centroids.push(data[first].clone());
    let mut closest_sq = vec![f32::INFINITY; data.len()];
    for j in 0..data.len() {
        closest_sq[j] = sq_l2(&data[j], &centroids[0]);
    }
    for _ in 1..k {
        let total: f32 = closest_sq.iter().sum();
        let target: f32 = rng.gen::<f32>() * total;
        let mut acc = 0.0f32;
        let mut chosen = data.len() - 1;
        for (j, &v) in closest_sq.iter().enumerate() {
            acc += v;
            if acc >= target {
                chosen = j;
                break;
            }
        }
        centroids.push(data[chosen].clone());
        let ci = centroids.len() - 1;
        for j in 0..data.len() {
            let dd = sq_l2(&data[j], &centroids[ci]);
            if dd < closest_sq[j] {
                closest_sq[j] = dd;
            }
        }
    }

    let mut assignments = vec![0u32; data.len()];
    for _ in 0..iters {
        // Assign
        for (j, x) in data.iter().enumerate() {
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for (c, cv) in centroids.iter().enumerate() {
                let dd = sq_l2(x, cv);
                if dd < best_d {
                    best_d = dd;
                    best = c as u32;
                }
            }
            assignments[j] = best;
        }
        // Update
        let mut sums = vec![vec![0.0f32; d]; k];
        let mut counts = vec![0u32; k];
        for (j, x) in data.iter().enumerate() {
            let c = assignments[j] as usize;
            counts[c] += 1;
            for t in 0..d {
                sums[c][t] += x[t];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for t in 0..d {
                    centroids[c][t] = sums[c][t] * inv;
                }
            } else {
                // Re-seed empty cluster to a random data point
                let idx = rng.gen_range(0..data.len());
                centroids[c] = data[idx].clone();
            }
        }
    }

    KMeansResult { centroids, assignments }
}
