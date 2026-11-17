//! Minimal Lloyd k-means with k-means++ seeding. Pure std, no SIMD —
//! we want the IMI vs flat-IVF comparison unaffected by inner-kernel
//! micro-optimizations.

use crate::data::Rng;

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// k-means++ seeding: picks one random point, then successively picks
/// points with probability proportional to squared distance to the
/// nearest chosen centroid.
fn kpp_seed(data: &[f32], d: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let n = data.len() / d;
    assert!(n >= k, "need at least k points for seeding");
    let mut centroids = Vec::<f32>::with_capacity(k * d);
    let first = rng.next_bounded(n as u32) as usize;
    centroids.extend_from_slice(&data[first * d..(first + 1) * d]);

    let mut best_sq = vec![f32::INFINITY; n];
    for c in 1..k {
        // Update best_sq against the most recently added centroid.
        let last = &centroids[(c - 1) * d..c * d];
        let mut total = 0.0f64;
        for i in 0..n {
            let dsq = l2_sq(&data[i * d..(i + 1) * d], last);
            if dsq < best_sq[i] {
                best_sq[i] = dsq;
            }
            total += best_sq[i] as f64;
        }
        // Weighted pick.
        let t = (rng.next_f32() as f64) * total;
        let mut acc = 0.0f64;
        let mut pick = n - 1;
        for i in 0..n {
            acc += best_sq[i] as f64;
            if acc >= t {
                pick = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[pick * d..(pick + 1) * d]);
    }
    centroids
}

/// Runs `iters` Lloyd iterations. Returns (centroids, assignments).
pub fn kmeans(data: &[f32], d: usize, k: usize, iters: usize, rng: &mut Rng) -> (Vec<f32>, Vec<u32>) {
    let n = data.len() / d;
    let mut centroids = kpp_seed(data, d, k, rng);
    let mut assign = vec![0u32; n];
    let mut sum = vec![0.0f64; k * d];
    let mut count = vec![0u32; k];

    for _it in 0..iters {
        // Assign step.
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_sq = f32::INFINITY;
            for c in 0..k {
                let sq = l2_sq(x, &centroids[c * d..(c + 1) * d]);
                if sq < best_sq {
                    best_sq = sq;
                    best = c;
                }
            }
            assign[i] = best as u32;
        }

        // Update step.
        for v in sum.iter_mut() {
            *v = 0.0;
        }
        for v in count.iter_mut() {
            *v = 0;
        }
        for i in 0..n {
            let a = assign[i] as usize;
            count[a] += 1;
            let x = &data[i * d..(i + 1) * d];
            let out = &mut sum[a * d..(a + 1) * d];
            for j in 0..d {
                out[j] += x[j] as f64;
            }
        }
        for c in 0..k {
            if count[c] == 0 {
                // Dead centroid: re-seed to a random point to avoid loss of
                // representational capacity (common guard for Lloyd).
                let r = rng.next_bounded(n as u32) as usize;
                let src = &data[r * d..(r + 1) * d];
                for j in 0..d {
                    centroids[c * d + j] = src[j];
                }
                continue;
            }
            let inv = 1.0f64 / count[c] as f64;
            let out = &mut centroids[c * d..(c + 1) * d];
            let s = &sum[c * d..(c + 1) * d];
            for j in 0..d {
                out[j] = (s[j] * inv) as f32;
            }
        }
    }
    (centroids, assign)
}
