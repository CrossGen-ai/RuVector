//! Deterministic k-means++ for centroid seeding, followed by a small number of
//! Lloyd iterations. Zero-dependency; uses [`Xorshift64`] for reproducibility.

use crate::rng::Xorshift64;
use crate::vec_math::l2_sq;

/// K-means configuration.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of centroids (partitions).
    pub k: usize,
    /// Number of Lloyd refinement iterations after k-means++ seeding.
    pub iterations: usize,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self { k: 16, iterations: 10, seed: 0xA57E_C0DE }
    }
}

/// Trained k-means model.
#[derive(Debug, Clone)]
pub struct KMeansModel {
    /// Centroid vectors, row-major, `k` rows of `dim` columns.
    pub centroids: Vec<f32>,
    /// Dimensionality.
    pub dim: usize,
    /// Number of centroids.
    pub k: usize,
}

impl KMeansModel {
    /// Slice view of centroid `i`.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        &self.centroids[i * self.dim..(i + 1) * self.dim]
    }

    /// Find the nearest centroid to `x` and its squared distance.
    pub fn nearest(&self, x: &[f32]) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for i in 0..self.k {
            let d = l2_sq(x, self.centroid(i));
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        (best, best_d)
    }

    /// Return centroid indices sorted by ascending distance to `x`, with the
    /// squared distance. Returns `min(top, k)` entries.
    pub fn top_nearest(&self, x: &[f32], top: usize) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> =
            (0..self.k).map(|i| (i, l2_sq(x, self.centroid(i)))).collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        all.truncate(top.min(self.k));
        all
    }
}

/// K-means++ seeding + Lloyd refinement.
///
/// `data` is a row-major `n × dim` float matrix. Returns a trained model.
///
/// # Panics
/// Panics if `data` is empty or its length is not a multiple of `dim`.
pub fn kmeans_pp(data: &[f32], dim: usize, cfg: &KMeansConfig) -> KMeansModel {
    assert!(!data.is_empty(), "kmeans_pp: empty data");
    assert!(dim > 0, "kmeans_pp: dim must be > 0");
    assert_eq!(data.len() % dim, 0, "kmeans_pp: data length not a multiple of dim");
    let n = data.len() / dim;
    let k = cfg.k.min(n).max(1);

    let mut rng = Xorshift64::new(cfg.seed);
    let mut centroids = vec![0.0f32; k * dim];

    // k-means++ seed 1: uniform random point.
    let first = rng.next_usize(n);
    centroids[..dim].copy_from_slice(&data[first * dim..(first + 1) * dim]);

    // Subsequent seeds: pick with probability proportional to D^2 to nearest
    // existing centroid.
    let mut dists = vec![f32::INFINITY; n];
    for c in 1..k {
        // Update D^2 given centroid c-1.
        let cc = &centroids[(c - 1) * dim..c * dim];
        for i in 0..n {
            let d = l2_sq(&data[i * dim..(i + 1) * dim], cc);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        // Weighted sample.
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            // All points already coincident with an existing centroid.
            let idx = rng.next_usize(n);
            centroids[c * dim..(c + 1) * dim]
                .copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
            continue;
        }
        let mut target = (rng.next_f32() * total) as f64;
        let mut chosen = n - 1;
        for i in 0..n {
            target -= dists[i] as f64;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids[c * dim..(c + 1) * dim]
            .copy_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
    }

    // Lloyd iterations.
    let mut assignments = vec![0usize; n];
    let mut sums = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];
    for _iter in 0..cfg.iterations {
        // Assign.
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cc = &centroids[c * dim..(c + 1) * dim];
                let d = l2_sq(x, cc);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assignments[i] = best;
        }
        // Update.
        sums.iter_mut().for_each(|v| *v = 0.0);
        counts.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let src = &data[i * dim..(i + 1) * dim];
            let dst = &mut sums[c * dim..(c + 1) * dim];
            for j in 0..dim {
                dst[j] += src[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: re-seed to a random point.
                let idx = rng.next_usize(n);
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
            } else {
                let inv = 1.0f32 / counts[c] as f32;
                let src = &sums[c * dim..(c + 1) * dim];
                let dst = &mut centroids[c * dim..(c + 1) * dim];
                for j in 0..dim {
                    dst[j] = src[j] * inv;
                }
            }
        }
    }

    KMeansModel { centroids, dim, k }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_finds_two_clusters() {
        // Two Gaussian-ish blobs in 2D.
        let mut data = Vec::new();
        let mut rng = Xorshift64::new(1);
        for _ in 0..50 {
            data.push(0.0 + rng.next_signed_f32() * 0.1);
            data.push(0.0 + rng.next_signed_f32() * 0.1);
        }
        for _ in 0..50 {
            data.push(5.0 + rng.next_signed_f32() * 0.1);
            data.push(5.0 + rng.next_signed_f32() * 0.1);
        }
        let cfg = KMeansConfig { k: 2, iterations: 20, seed: 42 };
        let model = kmeans_pp(&data, 2, &cfg);
        // One centroid near (0,0), the other near (5,5).
        let c0 = model.centroid(0);
        let c1 = model.centroid(1);
        let d00 = (c0[0]).abs() + (c0[1]).abs();
        let d55 = (c0[0] - 5.0).abs() + (c0[1] - 5.0).abs();
        // Whichever centroid is c0, its counterpart c1 must be near the other blob.
        if d00 < d55 {
            assert!((c1[0] - 5.0).abs() < 0.5);
            assert!((c1[1] - 5.0).abs() < 0.5);
        } else {
            assert!((c1[0]).abs() < 0.5);
            assert!((c1[1]).abs() < 0.5);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let mut rng = Xorshift64::new(7);
        let data: Vec<f32> = (0..400).map(|_| rng.next_signed_f32()).collect();
        let cfg = KMeansConfig { k: 4, iterations: 5, seed: 123 };
        let a = kmeans_pp(&data, 4, &cfg);
        let b = kmeans_pp(&data, 4, &cfg);
        assert_eq!(a.centroids, b.centroids);
    }

    #[test]
    fn nearest_returns_valid() {
        let mut rng = Xorshift64::new(3);
        let data: Vec<f32> = (0..200).map(|_| rng.next_signed_f32()).collect();
        let cfg = KMeansConfig { k: 3, iterations: 5, seed: 1 };
        let model = kmeans_pp(&data, 4, &cfg);
        for i in 0..50 {
            let x = &data[i * 4..(i + 1) * 4];
            let (idx, d) = model.nearest(x);
            assert!(idx < 3);
            assert!(d.is_finite());
        }
    }
}
