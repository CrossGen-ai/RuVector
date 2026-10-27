//! Synthetic data: Gaussian mixtures with a known cluster structure
//! (so per-cluster SQ has a clean story to tell).

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::kmeans::l2_sq;

/// Generate `n` points of dimension `d` drawn from `n_components` isotropic
/// Gaussians placed on a lattice-like set of centers.
pub fn make_gaussian_mixture(n: usize, d: usize, n_components: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // sample component centers from Uniform[-5, 5]^d
    let mut centers = vec![0.0f32; n_components * d];
    for v in centers.iter_mut() {
        *v = rng.gen_range(-5.0..5.0);
    }
    let mut data = vec![0.0f32; n * d];
    let mut labels = vec![0u32; n];
    for i in 0..n {
        let c = rng.gen_range(0..n_components);
        labels[i] = c as u32;
        for j in 0..d {
            // isotropic Gaussian std 1.0 (sum-of-6 approx)
            let mut s = 0.0f32;
            for _ in 0..6 { s += rng.gen::<f32>() - 0.5; }
            data[i * d + j] = centers[c * d + j] + s;
        }
    }
    (data, labels)
}

/// Exact top-k by squared L2 against `data` (n x d).
pub fn exact_topk(data: &[f32], d: usize, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = data.len() / d;
    let mut scored: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, l2_sq(&data[i * d..(i + 1) * d], query)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}
