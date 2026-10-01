//! Deterministic synthetic corpus for tests and benches.
//!
//! A cluster model with Gaussian noise: `nc` clusters, uniform means in [-1,1],
//! per-vector draw: pick a cluster, add N(0, sigma). This gives a workload
//! where PQ actually earns its keep (centroid distances are informative),
//! yielding pivot pressure that prefix pruning can exploit.

use crate::codec::{Codebook, PqCodes, PqParams};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub fn make_corpus(
    n: usize,
    m: usize,
    d: usize,
    nc: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Codebook, PqCodes) {
    assert!(d % m == 0);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut centers = vec![0.0f32; nc * d];
    for v in centers.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }
    let sigma = 0.15f32;
    let mut vecs = vec![0.0f32; n * d];
    for i in 0..n {
        let c = rng.gen_range(0..nc);
        for j in 0..d {
            let g: f32 = sample_normal(&mut rng, sigma);
            vecs[i * d + j] = centers[c * d + j] + g;
        }
    }
    // 32 queries drawn from the same distribution so recall is non-trivial.
    let mut queries = vec![0.0f32; 32 * d];
    for i in 0..32 {
        let c = rng.gen_range(0..nc);
        for j in 0..d {
            let g: f32 = sample_normal(&mut rng, sigma);
            queries[i * d + j] = centers[c * d + j] + g;
        }
    }

    let params = PqParams::new(d, m);
    let book = Codebook::train(&vecs, n, params, 10, seed ^ 0x5a5a_a5a5);
    let codes = PqCodes::encode(&vecs, n, &book);
    (vecs, queries, book, codes)
}

fn sample_normal(rng: &mut SmallRng, sigma: f32) -> f32 {
    // Box-Muller.
    let u1: f32 = rng.gen_range(1e-7f32..1.0);
    let u2: f32 = rng.gen_range(0.0f32..1.0);
    let r: f32 = (-2.0f32 * u1.ln()).sqrt();
    let theta: f32 = 2.0 * std::f32::consts::PI * u2;
    sigma * r * theta.cos()
}
