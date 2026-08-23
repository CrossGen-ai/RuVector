//! Minimal Lloyd's-algorithm k-means (batch, cosine-cost-agnostic, L2 only).
//! Purposefully simple so all three backends share identical centroids.

use crate::{sq_l2, rng::Xor64};

/// Train `k` centroids with `iters` Lloyd iterations. Deterministic given seed.
pub fn train(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(!data.is_empty() && k > 0 && k <= data.len());
    let dim = data[0].len();
    let mut rng = Xor64::new(seed);

    // k-means++ init: pick first center randomly, subsequent by D²-weighting.
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = (rng.next_u64() as usize) % data.len();
    centers.push(data[first].clone());
    let mut d2 = vec![f32::INFINITY; data.len()];
    while centers.len() < k {
        let last = centers.last().unwrap();
        let mut total = 0f32;
        for i in 0..data.len() {
            let d = sq_l2(&data[i], last);
            if d < d2[i] { d2[i] = d; }
            total += d2[i];
        }
        // Weighted sample
        let mut target = (rng.uniform() as f32) * total;
        let mut chosen = 0usize;
        for i in 0..data.len() {
            target -= d2[i];
            if target <= 0.0 { chosen = i; break; }
        }
        centers.push(data[chosen].clone());
    }

    // Lloyd iterations.
    let mut assign = vec![0usize; data.len()];
    for _ in 0..iters {
        // Assign
        for (i, v) in data.iter().enumerate() {
            let mut best = (f32::INFINITY, 0usize);
            for (c, cen) in centers.iter().enumerate() {
                let d = sq_l2(v, cen);
                if d < best.0 { best = (d, c); }
            }
            assign[i] = best.1;
        }
        // Update
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in data.iter().enumerate() {
            let a = assign[i];
            for d in 0..dim { sums[a][d] += v[d]; }
            counts[a] += 1;
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty cluster to a random point.
                let r = (rng.next_u64() as usize) % data.len();
                centers[c] = data[r].clone();
            } else {
                for d in 0..dim {
                    centers[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }
    centers
}
