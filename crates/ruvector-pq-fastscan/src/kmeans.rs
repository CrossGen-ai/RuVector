//! Minimal Lloyd's k-means in a single subspace.
//!
//! Used only for PQ codebook training. Not a public API.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Train `k` centroids for a flat list of `n` points of dimension `d`.
/// Returns `k * d` flat centroids. Deterministic for a given seed.
pub fn kmeans(points: &[f32], n: usize, d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    assert_eq!(points.len(), n * d);
    assert!(k > 0 && k <= n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![0f32; k * d];

    // Init: pick k distinct random points (sampling without replacement, simple shuffle prefix).
    let mut perm: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        perm.swap(i, j);
        let src = &points[perm[i] * d..(perm[i] + 1) * d];
        centroids[i * d..(i + 1) * d].copy_from_slice(src);
    }

    let mut assign = vec![0u32; n];
    let mut sums = vec![0f32; k * d];
    let mut counts = vec![0u32; k];

    for _ in 0..iters {
        // Assign
        for p in 0..n {
            let pt = &points[p * d..(p + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cc = &centroids[c * d..(c + 1) * d];
                let mut s = 0f32;
                for i in 0..d {
                    let dd = pt[i] - cc[i];
                    s += dd * dd;
                    if s >= best_d { break; }
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            assign[p] = best as u32;
        }

        // Update
        for v in sums.iter_mut() { *v = 0.0; }
        for v in counts.iter_mut() { *v = 0; }
        for p in 0..n {
            let c = assign[p] as usize;
            counts[c] += 1;
            let pt = &points[p * d..(p + 1) * d];
            let acc = &mut sums[c * d..(c + 1) * d];
            for i in 0..d { acc[i] += pt[i]; }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Reseed empty cluster from a random point.
                let r = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d]
                    .copy_from_slice(&points[r * d..(r + 1) * d]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for i in 0..d {
                    centroids[c * d + i] = sums[c * d + i] * inv;
                }
            }
        }
    }

    centroids
}

/// Encode each point to its nearest centroid index.
pub fn assign_nearest(points: &[f32], n: usize, d: usize, centroids: &[f32], k: usize) -> Vec<u8> {
    assert_eq!(points.len(), n * d);
    assert_eq!(centroids.len(), k * d);
    assert!(k <= 256, "assign_nearest returns u8; k must be <= 256");

    let mut out = vec![0u8; n];
    for p in 0..n {
        let pt = &points[p * d..(p + 1) * d];
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let cc = &centroids[c * d..(c + 1) * d];
            let mut s = 0f32;
            for i in 0..d {
                let dd = pt[i] - cc[i];
                s += dd * dd;
            }
            if s < best_d {
                best_d = s;
                best = c;
            }
        }
        out[p] = best as u8;
    }
    out
}
