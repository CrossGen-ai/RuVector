//! Deterministic dataset & query generators used by benches, examples, tests.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate a reproducible Gaussian-mixture dataset. `seed` is deterministic
/// across platforms.
pub fn gen_dataset(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = &centres[i % clusters];
        for j in 0..dim {
            let noise: f32 = rng.gen_range(-1.0..1.0);
            out[i * dim + j] = c[j] + noise;
        }
    }
    out
}

/// Generate `q` reproducible queries.
pub fn gen_queries(q: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0f32; q * dim];
    for v in out.iter_mut() {
        *v = rng.gen_range(-5.5..5.5);
    }
    out
}
