use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    // Box-Muller
    let u1: f32 = rng.gen_range(1e-7_f32..1.0);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn gaussian_dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = vec![0.0f32; d];
        for j in 0..d { v[j] = gauss(&mut rng); }
        out.push(v);
    }
    out
}

pub fn uniform_queries(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    gaussian_dataset(n, d, seed ^ 0xDEADBEEF)
}
