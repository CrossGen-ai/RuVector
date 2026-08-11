//! Minimal Lloyd's k-means, deterministic k-means++ init.

use crate::{sqdist, Rng};

pub struct KMeans {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<u32>,
}

impl KMeans {
    pub fn fit(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Self {
        assert!(!data.is_empty() && k > 0);
        let d = data[0].len();
        let mut rng = Rng::new(seed);
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        // k-means++ init
        centroids.push(data[rng.next_usize(data.len())].clone());
        let mut d2: Vec<f32> = data.iter().map(|v| sqdist(v, &centroids[0])).collect();
        for _ in 1..k {
            let sum: f32 = d2.iter().sum();
            if sum <= 0.0 {
                centroids.push(data[rng.next_usize(data.len())].clone());
            } else {
                let t = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * sum;
                let mut acc = 0f32;
                let mut pick = data.len() - 1;
                for (i, &w) in d2.iter().enumerate() {
                    acc += w;
                    if acc >= t {
                        pick = i;
                        break;
                    }
                }
                centroids.push(data[pick].clone());
                let last = centroids.last().unwrap();
                for (i, v) in data.iter().enumerate() {
                    let nd = sqdist(v, last);
                    if nd < d2[i] {
                        d2[i] = nd;
                    }
                }
            }
        }
        // Lloyd iterations
        let mut assign = vec![0u32; data.len()];
        for _ in 0..iters {
            for (i, v) in data.iter().enumerate() {
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for (c, cv) in centroids.iter().enumerate() {
                    let dd = sqdist(v, cv);
                    if dd < best_d {
                        best_d = dd;
                        best = c as u32;
                    }
                }
                assign[i] = best;
            }
            let mut sums = vec![vec![0f32; d]; k];
            let mut counts = vec![0u32; k];
            for (i, v) in data.iter().enumerate() {
                let c = assign[i] as usize;
                counts[c] += 1;
                for j in 0..d {
                    sums[c][j] += v[j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for j in 0..d {
                        centroids[c][j] = sums[c][j] * inv;
                    }
                }
            }
        }
        Self { centroids, assignments: assign }
    }

    /// Return the closest centroid indices to `q`, sorted by distance.
    pub fn nearest_centroids(&self, q: &[f32], m: usize) -> Vec<usize> {
        let mut pairs: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (sqdist(c, q), i))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        pairs.into_iter().take(m).map(|(_, i)| i).collect()
    }
}
