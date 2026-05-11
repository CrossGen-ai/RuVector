//! Minimal Lloyd k-means with k-means++ seeding. Deterministic given a seed.

use crate::distance::l2_sq;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct KMeansConfig {
    pub k: usize,
    pub iters: usize,
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self { k: 64, iters: 20, seed: 0xC0FFEE }
    }
}

/// Train centroids on `data` (n × d, row-major). Returns flat `k*d` centroid buffer.
pub fn kmeans_lloyd(data: &[f32], n: usize, d: usize, cfg: &KMeansConfig) -> Vec<f32> {
    assert_eq!(data.len(), n * d);
    assert!(cfg.k > 0 && cfg.k <= n);
    let k = cfg.k;
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    // ---- k-means++ seeding ----
    let mut centroids: Vec<f32> = vec![0.0; k * d];
    let first = rng.gen_range(0..n);
    centroids[..d].copy_from_slice(&data[first * d..(first + 1) * d]);

    let mut min_d2: Vec<f32> = (0..n)
        .map(|i| l2_sq(&data[i * d..(i + 1) * d], &centroids[..d]))
        .collect();

    for c in 1..k {
        let sum: f32 = min_d2.iter().sum();
        if sum <= 0.0 {
            // duplicate any point; degenerate dataset
            let idx = rng.gen_range(0..n);
            centroids[c * d..(c + 1) * d]
                .copy_from_slice(&data[idx * d..(idx + 1) * d]);
        } else {
            let mut pick = rng.gen::<f32>() * sum;
            let mut chosen = n - 1;
            for (i, &w) in min_d2.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids[c * d..(c + 1) * d]
                .copy_from_slice(&data[chosen * d..(chosen + 1) * d]);
        }
        let new_c = &centroids[c * d..(c + 1) * d];
        for i in 0..n {
            let dd = l2_sq(&data[i * d..(i + 1) * d], new_c);
            if dd < min_d2[i] {
                min_d2[i] = dd;
            }
        }
    }

    // ---- Lloyd iterations ----
    let mut assign = vec![0usize; n];
    for _ in 0..cfg.iters {
        // Assign step
        for i in 0..n {
            let mut best = 0usize;
            let mut bestd = f32::INFINITY;
            for c in 0..k {
                let dd = l2_sq(
                    &data[i * d..(i + 1) * d],
                    &centroids[c * d..(c + 1) * d],
                );
                if dd < bestd {
                    bestd = dd;
                    best = c;
                }
            }
            assign[i] = best;
        }
        // Update step
        let mut counts = vec![0u32; k];
        let mut sums = vec![0.0f32; k * d];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let row = &data[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c * d + j] += row[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // re-seed empty centroid to a random data point
                let idx = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d]
                    .copy_from_slice(&data[idx * d..(idx + 1) * d]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_clusters() {
        // Two well-separated clusters around (0,0) and (10,10).
        let mut data = Vec::new();
        for i in 0..50 {
            let t = i as f32 * 0.01;
            data.extend_from_slice(&[t, -t]);
        }
        for i in 0..50 {
            let t = i as f32 * 0.01;
            data.extend_from_slice(&[10.0 + t, 10.0 - t]);
        }
        let cfg = KMeansConfig { k: 2, iters: 10, seed: 7 };
        let c = kmeans_lloyd(&data, 100, 2, &cfg);
        // One centroid should be near (0,0), other near (10,10).
        let c0 = (c[0], c[1]);
        let c1 = (c[2], c[3]);
        let near_origin = (c0.0.abs() < 1.0 && c0.1.abs() < 1.0)
            || (c1.0.abs() < 1.0 && c1.1.abs() < 1.0);
        let near_ten = (c0.0 - 10.0).abs() < 1.0 || (c1.0 - 10.0).abs() < 1.0;
        assert!(near_origin && near_ten, "centroids: {:?}", c);
    }
}
