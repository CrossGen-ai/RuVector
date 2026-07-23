//! Synthetic data generation for benchmarks (no external corpora required).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate `n` d-dim vectors as a mixture of `clusters` Gaussians on the
/// unit-ish sphere. Returns (data, cluster_ids). Row-major, len == n*d.
pub fn synthetic_gaussian(n: usize, d: usize, clusters: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Cluster centers
    let mut centers = vec![0f32; clusters * d];
    for c in centers.iter_mut() {
        *c = rng.gen_range(-1.0..1.0);
    }
    // Normalize centers to have varied magnitudes
    for c in 0..clusters {
        let s: f32 = centers[c * d..(c + 1) * d].iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = 1.0 / s.max(1e-9);
        for v in &mut centers[c * d..(c + 1) * d] {
            *v *= scale;
        }
    }
    let mut data = vec![0f32; n * d];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = rng.gen_range(0..clusters);
        ids[i] = c as u32;
        for j in 0..d {
            let noise: f32 = box_muller(&mut rng) * 0.25;
            data[i * d + j] = centers[c * d + j] + noise;
        }
    }
    (data, ids)
}

fn box_muller(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-9..1.0);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// L2-normalize each d-vector (row-major).
pub fn l2_normalize(data: &[f32], d: usize) -> Vec<f32> {
    let n = data.len() / d;
    let mut out = data.to_vec();
    for i in 0..n {
        let s: f32 = out[i * d..(i + 1) * d].iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = 1.0 / s.max(1e-12);
        for v in &mut out[i * d..(i + 1) * d] {
            *v *= scale;
        }
    }
    out
}

/// Deterministically sample `count` vectors from `data` as queries (with jitter).
pub fn sample(data: &[f32], d: usize, count: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / d;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count * d);
    for _ in 0..count {
        let i = rng.gen_range(0..n);
        for j in 0..d {
            let jitter = box_muller(&mut rng) * 0.05;
            out.push(data[i * d + j] + jitter);
        }
    }
    out
}
