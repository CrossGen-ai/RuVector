//! Small utilities: recall@k computation and skewed-variance data generation.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn gauss_skewed(n: usize, d: usize, seed: u64, skew_ratio: f32) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|i| {
                    let u1: f32 = rng.gen::<f32>().max(1e-9);
                    let u2: f32 = rng.gen::<f32>();
                    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    if i < d / 2 { g * skew_ratio } else { g / skew_ratio }
                })
                .collect()
        })
        .collect()
}

pub fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (crate::sq_dist(q, v), i))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

pub fn recall_at_k(gt: &[usize], approx: &[usize], k: usize) -> f32 {
    let mut hits = 0;
    for i in 0..k {
        if gt.contains(&approx[i]) { hits += 1; }
    }
    hits as f32 / k as f32
}
