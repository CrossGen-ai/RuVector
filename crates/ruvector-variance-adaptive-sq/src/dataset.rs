//! Synthetic dataset generator. Real OSS embeddings are heavy-tailed per
//! dimension: a few dimensions dominate the variance. We mimic that with a
//! log-normal variance profile and independent Gaussian samples per dim.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct Dataset {
    pub dim: usize,
    pub train: Vec<f32>,
    pub base: Vec<f32>,
    pub queries: Vec<f32>,
    pub variances: Vec<f32>,
}

/// Generate `n_train + n_base + n_query` vectors of dim `dim`. Dim `j` has
/// standard deviation `sigma_j` drawn once from a log-normal(mu, sigma_log).
pub fn gen_lognormal_embeddings(
    dim: usize,
    n_train: usize,
    n_base: usize,
    n_query: usize,
    sigma_log: f32,
    seed: u64,
) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sigmas = Vec::with_capacity(dim);
    for _ in 0..dim {
        let u1: f32 = rng.gen::<f32>().max(1e-9);
        let u2: f32 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let sigma = (sigma_log * z).exp();
        sigmas.push(sigma);
    }

    let total = n_train + n_base + n_query;
    let mut flat = Vec::with_capacity(total * dim);
    for _ in 0..total {
        for j in 0..dim {
            let u1: f32 = rng.gen::<f32>().max(1e-9);
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            flat.push(z * sigmas[j]);
        }
    }

    let train = flat[..n_train * dim].to_vec();
    let base = flat[n_train * dim..(n_train + n_base) * dim].to_vec();
    let queries = flat[(n_train + n_base) * dim..].to_vec();

    let variances: Vec<f32> = sigmas.iter().map(|s| s * s).collect();

    Dataset {
        dim,
        train,
        base,
        queries,
        variances,
    }
}
