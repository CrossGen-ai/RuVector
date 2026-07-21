//! Minimal Lloyd's k-means used to train per-subspace codebooks.
//!
//! Deliberately small: k-means++ init, a hard cap on iterations, and a
//! configurable min-cluster fallback that re-seeds an empty centroid on
//! the currently-worst-fit point. No SIMD, no mini-batch — the crate is
//! benchmark-honest, not an ANN production library.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// One trained k-means run.
pub struct KMeansResult {
    /// `k * dim` centroid buffer laid out row-major.
    pub centroids: Vec<f32>,
    /// Sum of squared distances from every point to its assigned centroid.
    pub distortion: f64,
    /// Iterations actually used (may be less than `max_iters`).
    pub iters: usize,
}

/// Train `k` centroids over `points` (each of length `dim`) using
/// k-means++ seeding. Returns the trained codebook plus the final
/// distortion sum.
pub fn kmeans(
    points: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    max_iters: usize,
    seed: u64,
) -> KMeansResult {
    assert_eq!(points.len(), n * dim);
    assert!(k > 0 && k <= n, "k must be in 1..=n");

    let mut rng = StdRng::seed_from_u64(seed);

    // k-means++ seeding.
    let mut centroids = vec![0f32; k * dim];
    let first = rng.gen_range(0..n);
    centroids[0..dim].copy_from_slice(&points[first * dim..(first + 1) * dim]);

    let mut best_sq = vec![f32::INFINITY; n];
    for c in 0..k {
        // Update best_sq w.r.t. centroid c-1 (for c==0 recompute from centroid 0).
        let center = &centroids[c.saturating_sub(0) * dim..(c.saturating_sub(0) + 1) * dim];
        let effective = if c == 0 { 0 } else { c - 1 };
        let center = &centroids[effective * dim..(effective + 1) * dim];
        for i in 0..n {
            let d = sq_l2(&points[i * dim..(i + 1) * dim], center);
            if d < best_sq[i] {
                best_sq[i] = d;
            }
        }
        if c + 1 == k {
            break;
        }
        // Sample next centroid weighted by best_sq.
        let sum: f64 = best_sq.iter().map(|v| *v as f64).sum();
        if sum <= 0.0 {
            // Everything is duplicate — pick uniformly.
            let idx = rng.gen_range(0..n);
            centroids[(c + 1) * dim..(c + 2) * dim]
                .copy_from_slice(&points[idx * dim..(idx + 1) * dim]);
            continue;
        }
        let mut target = rng.gen::<f64>() * sum;
        let mut chosen = n - 1;
        for i in 0..n {
            target -= best_sq[i] as f64;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids[(c + 1) * dim..(c + 2) * dim]
            .copy_from_slice(&points[chosen * dim..(chosen + 1) * dim]);
    }

    // Lloyd's iterations.
    let mut assignments = vec![0u32; n];
    let mut prev_distortion = f64::INFINITY;
    let mut iters_used = 0usize;

    for it in 0..max_iters.max(1) {
        iters_used = it + 1;
        let mut distortion = 0f64;
        for i in 0..n {
            let point = &points[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = sq_l2(point, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assignments[i] = best as u32;
            distortion += best_d as f64;
        }

        // Recompute centroids as the mean of assigned points, re-seeding
        // any empty cluster on the worst-fit point to avoid dead codes.
        let mut sums = vec![0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let row = &points[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums[c * dim + d] += row[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty cluster on the current worst-fit point.
                let mut worst = 0usize;
                let mut worst_d = -1f32;
                for i in 0..n {
                    let a = assignments[i] as usize;
                    let d = sq_l2(
                        &points[i * dim..(i + 1) * dim],
                        &centroids[a * dim..(a + 1) * dim],
                    );
                    if d > worst_d {
                        worst_d = d;
                        worst = i;
                    }
                }
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&points[worst * dim..(worst + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }

        // Convergence check.
        if (prev_distortion - distortion).abs() < 1e-6 * prev_distortion.max(1.0) {
            prev_distortion = distortion;
            break;
        }
        prev_distortion = distortion;
    }

    // Silence unused-warning safety net for `rng` when max_iters==0.
    let _ = rng.gen::<u8>();
    let mut tmp = [0u8; 0];
    tmp.shuffle(&mut StdRng::seed_from_u64(seed));

    KMeansResult {
        centroids,
        distortion: prev_distortion,
        iters: iters_used,
    }
}

/// Squared L2 distance between two equal-length slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}
