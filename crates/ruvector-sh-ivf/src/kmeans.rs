//! Tiny Lloyd k-means for IVF centroids and PQ subspaces.
//!
//! Not production grade — fixed iteration count, no mini-batch, no reseeding.
//! Sufficient for a research-scale benchmark on 50k x 96 floats.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::l2sq;

pub struct Kmeans {
    pub d: usize,
    pub k: usize,
    pub centroids: Vec<f32>,
}

impl Kmeans {
    pub fn train(base: &[f32], d: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert_eq!(base.len() % d, 0);
        let n = base.len() / d;
        assert!(n >= k, "need at least k vectors to seed centroids");

        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        // k-means++ init.
        let mut centroids = Vec::with_capacity(k * d);
        let first = rng.gen_range(0..n);
        centroids.extend_from_slice(&base[first * d..(first + 1) * d]);
        let mut dists = vec![f32::INFINITY; n];
        for _ in 1..k {
            for i in 0..n {
                let last_c = &centroids[(centroids.len() - d)..];
                let dist = l2sq(&base[i * d..(i + 1) * d], last_c);
                if dist < dists[i] {
                    dists[i] = dist;
                }
            }
            let total: f32 = dists.iter().sum::<f32>().max(1e-12);
            let r = rng.gen::<f32>() * total;
            let mut acc = 0.0f32;
            let mut pick = n - 1;
            for i in 0..n {
                acc += dists[i];
                if acc >= r {
                    pick = i;
                    break;
                }
            }
            centroids.extend_from_slice(&base[pick * d..(pick + 1) * d]);
        }

        let mut assignments = vec![0u32; n];
        let mut sums = vec![0f32; k * d];
        let mut counts = vec![0u32; k];
        for _ in 0..iters {
            // Assign.
            for i in 0..n {
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let dist = l2sq(&base[i * d..(i + 1) * d], &centroids[c * d..(c + 1) * d]);
                    if dist < best_d {
                        best_d = dist;
                        best = c;
                    }
                }
                assignments[i] = best as u32;
            }
            // Update.
            sums.fill(0.0);
            counts.fill(0);
            for i in 0..n {
                let c = assignments[i] as usize;
                counts[c] += 1;
                for j in 0..d {
                    sums[c * d + j] += base[i * d + j];
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    continue;
                }
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            }
        }

        Kmeans { d, k, centroids }
    }

    #[inline]
    pub fn nearest(&self, x: &[f32]) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let dist = l2sq(x, &self.centroids[c * self.d..(c + 1) * self.d]);
            if dist < best_d {
                best_d = dist;
                best = c;
            }
        }
        (best, best_d)
    }
}
