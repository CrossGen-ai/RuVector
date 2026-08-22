//! Minimal Lloyd's k-means for training RVQ codebooks.
//!
//! Deterministic given a fixed seed. Uses k-means++ seeding for stability
//! at small cluster counts (K=16..256).

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Train `k` centroids on `data` (row-major, `n` rows of `d` cols).
/// Returns centroids as `k * d` row-major `Vec<f32>`.
///
/// Panics if `data.len() != n * d` or `n < k`.
pub fn kmeans(data: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    assert_eq!(data.len(), n * d, "data length must be n*d");
    assert!(n >= k, "need at least k points to train k clusters");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeanspp_seed(data, n, d, k, &mut rng);
    let mut assign = vec![0u32; n];

    for _ in 0..iters {
        // 1. assignment step
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cc = &centroids[c * d..(c + 1) * d];
                let dist = l2_sq(x, cc);
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }

        // 2. update step
        let mut sums = vec![0f32; k * d];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let dst = &mut sums[c * d..(c + 1) * d];
            let src = &data[i * d..(i + 1) * d];
            for j in 0..d {
                dst[j] += src[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // reseed empty cluster with a random point
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

fn kmeanspp_seed(data: &[f32], n: usize, d: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * d);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&data[first * d..(first + 1) * d]);

    let mut min_d2 = vec![f32::INFINITY; n];
    for i in 0..n {
        min_d2[i] = l2_sq(&data[i * d..(i + 1) * d], &centroids[..d]);
    }

    for c in 1..k {
        let total: f32 = min_d2.iter().sum();
        if total <= 0.0 {
            // duplicated data — pick random
            let idx = rng.gen_range(0..n);
            centroids.extend_from_slice(&data[idx * d..(idx + 1) * d]);
            continue;
        }
        let t: f32 = rng.gen_range(0.0..total);
        let mut acc = 0f32;
        let mut chosen = n - 1;
        for i in 0..n {
            acc += min_d2[i];
            if acc >= t {
                chosen = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[chosen * d..(chosen + 1) * d]);
        // update min_d2
        let cc = &centroids[c * d..(c + 1) * d].to_vec();
        for i in 0..n {
            let dist = l2_sq(&data[i * d..(i + 1) * d], cc);
            if dist < min_d2[i] {
                min_d2[i] = dist;
            }
        }
    }
    centroids
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_converges_on_separable_blobs() {
        // 3 well-separated 1-D clusters
        let mut data = Vec::new();
        for _ in 0..30 { data.push(0.0); }
        for _ in 0..30 { data.push(10.0); }
        for _ in 0..30 { data.push(20.0); }
        let cents = kmeans(&data, 90, 1, 3, 20, 42);
        let mut c: Vec<f32> = cents.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((c[0] - 0.0).abs() < 0.5);
        assert!((c[1] - 10.0).abs() < 0.5);
        assert!((c[2] - 20.0).abs() < 0.5);
    }
}
