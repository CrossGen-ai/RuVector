//! Tiny Lloyd k-means used to train PQ subquantizer codebooks.
//!
//! No external BLAS — works on slices. Designed for small `d` (subspace
//! dimension `d/m`, typically 4–16) and `k = 256`, so cost stays modest even
//! at hundreds of thousands of training rows.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Run Lloyd's algorithm. Returns row-major `k × d` centroids.
///
/// * `data` — `n × d` row-major.
/// * `iters` — Lloyd passes (10–25 is enough for small `d`).
/// * `seed` — for reproducible init.
pub fn kmeans(data: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    assert_eq!(data.len(), n * d);
    assert!(n >= k, "n={} must be >= k={}", n, k);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeanspp_init(data, n, d, k, &mut rng);
    let mut assignments = vec![0u32; n];

    for _ in 0..iters {
        // Assign each row to nearest centroid.
        let mut changed = 0usize;
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let mut best = 0u32;
            let mut bestd = f32::INFINITY;
            for c in 0..k {
                let cv = &centroids[c * d..(c + 1) * d];
                let dist = sql2(row, cv);
                if dist < bestd {
                    bestd = dist;
                    best = c as u32;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed += 1;
            }
        }

        // Recompute centroids as cluster means.
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let row = &data[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c * d + j] += row[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            } else {
                // Empty cluster: re-seed from random data row.
                let r = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[r * d..(r + 1) * d]);
            }
        }

        if changed == 0 {
            break;
        }
    }

    centroids
}

/// k-means++ seeding: probability ∝ d²(x, nearest chosen centroid).
fn kmeanspp_init(data: &[f32], n: usize, d: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut centroids = vec![0.0f32; k * d];
    // First centroid: uniform random row.
    let r0 = rng.gen_range(0..n);
    centroids[..d].copy_from_slice(&data[r0 * d..(r0 + 1) * d]);

    let mut dists = vec![f32::INFINITY; n];
    for c in 1..k {
        // Update min-dist² to currently chosen centroids.
        let last = &centroids[(c - 1) * d..c * d];
        let mut total = 0.0f64;
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let dd = sql2(row, last);
            if dd < dists[i] {
                dists[i] = dd;
            }
            total += dists[i] as f64;
        }
        if total <= 0.0 {
            // Degenerate (all identical): copy a random row.
            let r = rng.gen_range(0..n);
            centroids[c * d..(c + 1) * d].copy_from_slice(&data[r * d..(r + 1) * d]);
            continue;
        }
        // Sample proportional to dists[i].
        let mut t = rng.gen::<f64>() * total;
        let mut pick = n - 1;
        for i in 0..n {
            t -= dists[i] as f64;
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        centroids[c * d..(c + 1) * d].copy_from_slice(&data[pick * d..(pick + 1) * d]);
    }
    centroids
}

#[inline]
fn sql2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
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
    fn separates_well_separated_blobs() {
        // 3 tight blobs centred at (0,0), (10,0), (0,10).
        let mut data = Vec::new();
        let centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut rng = StdRng::seed_from_u64(7);
        for c in &centers {
            for _ in 0..30 {
                data.push(c[0] + rng.gen::<f32>() * 0.1);
                data.push(c[1] + rng.gen::<f32>() * 0.1);
            }
        }
        let centroids = kmeans(&data, 90, 2, 3, 25, 42);
        // Each true centre should be near *some* learned centroid.
        for c in &centers {
            let mut best = f32::INFINITY;
            for cc in 0..3 {
                let learned = &centroids[cc * 2..cc * 2 + 2];
                let dx = c[0] - learned[0];
                let dy = c[1] - learned[1];
                best = best.min(dx * dx + dy * dy);
            }
            assert!(best < 0.5, "no learned centroid near {:?}: best={}", c, best);
        }
    }
}
