//! Minimal Lloyd-style k-means with k-means++ seeding.
//! No external linear-algebra dep; small enough to audit.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::metrics::sq_l2;

pub struct KMeans {
    pub centroids: Vec<Vec<f32>>,
    pub dim: usize,
}

impl KMeans {
    /// Run k-means++ init + `iters` Lloyd iterations.
    pub fn fit(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Self {
        assert!(!data.is_empty(), "kmeans: empty data");
        let dim = data[0].len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = kmeans_pp_seed(data, k, &mut rng);

        let mut assign = vec![0usize; data.len()];
        for _ in 0..iters {
            // Assign
            for (i, x) in data.iter().enumerate() {
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (c, cv) in centroids.iter().enumerate() {
                    let d = sq_l2(x, cv);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assign[i] = best;
            }
            // Update
            let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
            let mut counts = vec![0u32; k];
            for (i, x) in data.iter().enumerate() {
                let c = assign[i];
                counts[c] += 1;
                for d in 0..dim {
                    sums[c][d] += x[d];
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    // Re-seed dead centroid from a random data point.
                    let r = rng.gen_range(0..data.len());
                    centroids[c] = data[r].clone();
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..dim {
                        centroids[c][d] = sums[c][d] * inv;
                    }
                }
            }
        }
        KMeans { centroids, dim }
    }

    pub fn k(&self) -> usize {
        self.centroids.len()
    }
}

fn kmeans_pp_seed(data: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut chosen: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    chosen.push(data[first].clone());

    let mut d2 = vec![f32::INFINITY; n];
    for _ in 1..k {
        for i in 0..n {
            let last = chosen.last().unwrap();
            let d = sq_l2(&data[i], last);
            if d < d2[i] {
                d2[i] = d;
            }
        }
        let sum: f32 = d2.iter().sum();
        if sum <= 0.0 {
            // All duplicates; just pick a random point.
            chosen.push(data[rng.gen_range(0..n)].clone());
            continue;
        }
        let mut r = rng.gen::<f32>() * sum;
        let mut idx = 0;
        for i in 0..n {
            r -= d2[i];
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        chosen.push(data[idx].clone());
    }
    chosen
}
