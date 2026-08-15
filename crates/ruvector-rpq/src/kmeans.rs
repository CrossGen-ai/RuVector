//! K-means (Lloyd's algorithm) with k-means++ seeding, used to train PQ
//! codebooks per subspace. Single-threaded, dependency-free.

use crate::rng::Rng;

/// Squared L2 between two equal-length slices. Panics if lengths differ.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Train `k` centroids on `data` of dim `dim` with `iters` Lloyd iterations.
///
/// `data.len()` must be a multiple of `dim`. `k` must be > 0 and <= n.
pub fn kmeans(data: &[f32], dim: usize, k: usize, iters: usize, rng: &mut Rng) -> Vec<f32> {
    let n = data.len() / dim;
    assert!(k > 0 && k <= n, "kmeans: bad k ({k}) for n={n}");
    // k-means++ style init: pick first at random, then farthest-point sampling.
    let mut centroids = vec![0.0f32; k * dim];
    let first = (rng.next_u64() as usize) % n;
    centroids[..dim].copy_from_slice(&data[first * dim..(first + 1) * dim]);
    let mut min_d2 = vec![f32::INFINITY; n];
    for c in 1..k {
        // Update min distance to already-picked centroids.
        let prev = &centroids[(c - 1) * dim..c * dim];
        for i in 0..n {
            let d = sq_l2(&data[i * dim..(i + 1) * dim], prev);
            if d < min_d2[i] {
                min_d2[i] = d;
            }
        }
        // Sample proportional to min_d2. Fall back to argmax if all zero.
        let sum: f64 = min_d2.iter().map(|&v| v as f64).sum();
        let pick = if sum <= 0.0 {
            (rng.next_u64() as usize) % n
        } else {
            let target = rng.next_f32() as f64 * sum;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for i in 0..n {
                acc += min_d2[i] as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[pick * dim..(pick + 1) * dim]);
    }

    let mut assign = vec![0u32; n];
    let mut sums = vec![0.0f32; k * dim];
    let mut counts = vec![0u32; k];

    for _ in 0..iters {
        // Assignment step.
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = sq_l2(x, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }
        // Update step.
        sums.iter_mut().for_each(|v| *v = 0.0);
        counts.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let a = assign[i] as usize;
            counts[a] += 1;
            let dst = &mut sums[a * dim..(a + 1) * dim];
            let src = &data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                dst[d] += src[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty cluster from a random point.
                let r = (rng.next_u64() as usize) % n;
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&data[r * dim..(r + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                let dst = &mut centroids[c * dim..(c + 1) * dim];
                let src = &sums[c * dim..(c + 1) * dim];
                for d in 0..dim {
                    dst[d] = src[d] * inv;
                }
            }
        }
    }
    centroids
}
