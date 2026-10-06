//! Lloyd-Max optimal 1-D quantizer training.
//!
//! Given samples `x` for one dimension, find 16 centroid levels that minimize
//! squared error. This is the classic Lloyd algorithm specialized to 1-D with
//! uniform init, which also happens to be k-means with k=16 on a scalar.
//!
//! The two update rules alternate:
//! - Boundary step: thresholds are midpoints between adjacent centroids.
//! - Centroid step: each centroid becomes the mean of its assigned bin.
//!
//! Returns centroid values sorted ascending. Empty bins keep their previous
//! centroid, which is standard for Lloyd-Max.

use crate::NLEVELS;

/// Train 16-level Lloyd-Max quantizer on 1-D samples.
///
/// `max_iters` typically ≤ 25 is enough. `tol` is relative MSE improvement.
pub fn fit_lloyd_max_1d(samples: &[f32], max_iters: usize, tol: f32) -> [f32; NLEVELS] {
    assert!(!samples.is_empty(), "need at least one sample");
    let mut sorted: Vec<f32> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(!sorted.is_empty(), "no finite samples");

    let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
    // Uniform init of centroids inside [lo, hi].
    let mut centroids = [0f32; NLEVELS];
    if (hi - lo).abs() < f32::EPSILON {
        // Degenerate dimension: all values identical. Centroids collapse.
        centroids.fill(lo);
        return centroids;
    }
    for k in 0..NLEVELS {
        let t = (k as f32 + 0.5) / NLEVELS as f32;
        centroids[k] = lo + t * (hi - lo);
    }

    let mut prev_mse = f32::INFINITY;
    for _ in 0..max_iters {
        // Boundary = midpoint between consecutive centroids.
        let mut boundaries = [0f32; NLEVELS - 1];
        for k in 0..NLEVELS - 1 {
            boundaries[k] = 0.5 * (centroids[k] + centroids[k + 1]);
        }

        // Partition sorted samples via binary search on boundaries.
        let mut sums = [0f64; NLEVELS];
        let mut counts = [0u64; NLEVELS];
        let mut mse = 0f64;

        // Walk a two-pointer: j tracks which boundary we're past.
        let mut j = 0usize;
        for &x in sorted.iter() {
            while j < NLEVELS - 1 && x > boundaries[j] {
                j += 1;
            }
            let d = (x - centroids[j]) as f64;
            mse += d * d;
            sums[j] += x as f64;
            counts[j] += 1;
            // reset j for next sample? No — samples are sorted so j is monotone.
            // But to be strictly correct we reset once we overshoot. Since x is
            // sorted ascending, j is monotone non-decreasing, so leaving it is fine.
        }
        mse /= sorted.len() as f64;

        // Centroid update: mean of bin, or keep previous if empty.
        for k in 0..NLEVELS {
            if counts[k] > 0 {
                centroids[k] = (sums[k] / counts[k] as f64) as f32;
            }
        }

        let improved = (prev_mse - mse as f32) / prev_mse.max(f32::EPSILON);
        prev_mse = mse as f32;
        if improved < tol {
            break;
        }
    }

    // Return sorted — centroids may have shifted but order is preserved.
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_uniform_reaches_known_distribution() {
        // Uniform(0,1) with Lloyd-Max: 16 centroids should be close to midpoints.
        let mut rng = fastrng(42);
        let data: Vec<f32> = (0..20_000).map(|_| rng.next_f32()).collect();
        let c = fit_lloyd_max_1d(&data, 50, 1e-5);
        // Monotone increasing.
        for w in c.windows(2) {
            assert!(w[1] >= w[0]);
        }
        // First and last roughly bracket the data.
        assert!(c[0] < 0.1 && c[15] > 0.9);
    }

    #[test]
    fn constant_dimension_is_handled() {
        let data = vec![1.5f32; 100];
        let c = fit_lloyd_max_1d(&data, 20, 1e-5);
        for v in c {
            assert!((v - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn gaussian_mse_better_than_uniform_init() {
        let mut rng = fastrng(7);
        // Standard normal samples via Box-Muller.
        let data: Vec<f32> = (0..40_000).map(|_| rng.std_normal()).collect();

        let (lo, hi) = {
            let mut s = data.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (s[0], s[s.len() - 1])
        };
        let uniform: Vec<f32> = (0..NLEVELS)
            .map(|k| lo + (hi - lo) * (k as f32 + 0.5) / NLEVELS as f32)
            .collect();
        let lloyd = fit_lloyd_max_1d(&data, 30, 1e-6);

        let mse_u = mse(&data, &uniform);
        let mse_l = mse(&data, &lloyd);
        assert!(
            mse_l < mse_u * 0.6,
            "lloyd should significantly beat uniform on Gaussian: {} vs {}",
            mse_l,
            mse_u
        );
    }

    fn mse(data: &[f32], centroids: &[f32]) -> f32 {
        let mut acc = 0f32;
        for &x in data {
            let mut best = f32::INFINITY;
            for &c in centroids {
                let d = (x - c).abs();
                if d < best {
                    best = d;
                }
            }
            acc += best * best;
        }
        acc / data.len() as f32
    }

    // Tiny deterministic PRNG so tests have no extra deps beyond rand/rand_distr.
    pub struct FastRng(pub u64);
    pub fn fastrng(seed: u64) -> FastRng {
        FastRng(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    impl FastRng {
        pub fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        pub fn std_normal(&mut self) -> f32 {
            let u1 = (self.next_f32().max(1e-9)).min(1.0 - 1e-9);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * std::f32::consts::PI * u2).cos()
        }
    }
}
