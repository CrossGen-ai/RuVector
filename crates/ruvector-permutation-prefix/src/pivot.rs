//! Pivot selection by farthest-point sampling (FPS).
//!
//! FPS picks a diverse set of reference vectors from the dataset without
//! running k-means. It gives good spatial coverage for permutation-prefix
//! encoding.

use crate::sq_l2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Returns `n_pivots` dataset indices chosen by farthest-point sampling.
///
/// Starts from a seeded random vector. Each subsequent pivot is the vector
/// farthest (in squared-L2) from the already-chosen set.
pub fn farthest_point_sample(vectors: &[Vec<f32>], n_pivots: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = vectors.len();
    assert!(n >= n_pivots, "need at least n_pivots vectors");
    let mut rng = StdRng::seed_from_u64(seed);
    let first = rng.gen_range(0..n);

    let mut picked_idx: Vec<usize> = vec![first];
    let mut min_dist: Vec<f32> = vectors
        .iter()
        .map(|v| sq_l2(v, &vectors[first]))
        .collect();

    while picked_idx.len() < n_pivots {
        let mut best_i = 0usize;
        let mut best_d = f32::NEG_INFINITY;
        for (i, &d) in min_dist.iter().enumerate() {
            if d > best_d {
                best_d = d;
                best_i = i;
            }
        }
        picked_idx.push(best_i);
        let newv = &vectors[best_i];
        for (i, v) in vectors.iter().enumerate() {
            let d = sq_l2(v, newv);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    picked_idx.iter().map(|&i| vectors[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_returns_requested_count_and_is_deterministic() {
        let vs: Vec<Vec<f32>> = (0..32)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        let a = farthest_point_sample(&vs, 8, 42);
        let b = farthest_point_sample(&vs, 8, 42);
        assert_eq!(a.len(), 8);
        assert_eq!(a, b);
    }

    #[test]
    fn fps_includes_extremes() {
        // Two clearly-separated clusters should produce pivots in both.
        let mut vs: Vec<Vec<f32>> = (0..20).map(|_| vec![0.01, 0.0]).collect();
        vs.extend((0..20).map(|_| vec![10.0, 10.0]));
        let pivots = farthest_point_sample(&vs, 2, 7);
        assert_eq!(pivots.len(), 2);
        // The second pivot should be ~10 from the first.
        let d = sq_l2(&pivots[0], &pivots[1]);
        assert!(d > 1.0, "FPS pivots should be far apart, got sqL2={d}");
    }
}
