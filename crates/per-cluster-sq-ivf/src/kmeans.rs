//! Minimal Lloyd k-means on dense float32 vectors.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub struct KMeansResult {
    pub centroids: Vec<f32>, // k * d
    pub assignments: Vec<u32>, // n
    pub k: usize,
    pub d: usize,
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        s += diff * diff;
    }
    s
}

/// Lloyd's algorithm. k-means++ seed, max_iter Lloyd steps.
pub fn kmeans(data: &[f32], n: usize, d: usize, k: usize, max_iter: usize, seed: u64) -> KMeansResult {
    assert_eq!(data.len(), n * d);
    assert!(k > 0 && k <= n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centroids = vec![0.0f32; k * d];

    // k-means++ init
    let first = rng.gen_range(0..n);
    centroids[0..d].copy_from_slice(&data[first * d..(first + 1) * d]);
    let mut d2 = vec![f32::INFINITY; n];
    for ci in 0..k.saturating_sub(1) {
        for i in 0..n {
            let dist = l2_sq(&data[i * d..(i + 1) * d], &centroids[ci * d..(ci + 1) * d]);
            if dist < d2[i] { d2[i] = dist; }
        }
        let total: f32 = d2.iter().copied().sum();
        if total <= 0.0 {
            // fall back to random pick
            let r = rng.gen_range(0..n);
            centroids[(ci + 1) * d..(ci + 2) * d].copy_from_slice(&data[r * d..(r + 1) * d]);
            continue;
        }
        let threshold = rng.gen::<f32>() * total;
        let mut cum = 0.0f32;
        let mut picked = n - 1;
        for i in 0..n {
            cum += d2[i];
            if cum >= threshold {
                picked = i; break;
            }
        }
        centroids[(ci + 1) * d..(ci + 2) * d].copy_from_slice(&data[picked * d..(picked + 1) * d]);
    }

    let mut assignments = vec![0u32; n];
    for _iter in 0..max_iter {
        // assign
        let mut changed = 0usize;
        for i in 0..n {
            let v = &data[i * d..(i + 1) * d];
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let dist = l2_sq(v, &centroids[c * d..(c + 1) * d]);
                if dist < best_d { best_d = dist; best = c as u32; }
            }
            if assignments[i] != best { changed += 1; }
            assignments[i] = best;
        }
        // update
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let base = c * d;
            let vb = i * d;
            for j in 0..d { sums[base + j] += data[vb + j]; }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // reseed empty cluster to a random point
                let r = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[r * d..(r + 1) * d]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d { centroids[c * d + j] = sums[c * d + j] * inv; }
            }
        }
        if changed == 0 { break; }
    }

    KMeansResult { centroids, assignments, k, d }
}

/// Assign query to nearest centroid; return top-nprobe cluster ids.
pub fn probe_clusters(centroids: &[f32], d: usize, k: usize, query: &[f32], nprobe: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..k)
        .map(|c| (l2_sq(query, &centroids[c * d..(c + 1) * d]), c as u32))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.truncate(nprobe);
    scored.into_iter().map(|(_, c)| c).collect()
}
