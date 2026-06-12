//! Synthetic dataset generator that produces realistic clustered embeddings.
//!
//! We sample C cluster centers uniformly on the unit hypersphere, then draw
//! points from a von-Mises-Fisher-like distribution (Gaussian noise on the
//! sphere) around each center. Queries are drawn from the same distribution
//! but with a different RNG seed. This gives a non-trivial recall curve
//! (some queries are easy, some are hard), unlike pure-uniform data.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// Cosine distance on L2-normalized vectors: 1 - dot.
#[inline]
pub fn cos_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

pub struct Dataset {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    /// Ground truth top-k indices per query.
    pub gt: Vec<Vec<u32>>,
    pub k: usize,
}

fn sample_sphere(rng: &mut Xoshiro256PlusPlus, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for x in v.iter_mut() {
        // Box-Muller
        let u1: f32 = rng.gen_range(1e-7..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        *x = r * t.cos();
    }
    normalize(&mut v);
    v
}

fn perturb(rng: &mut Xoshiro256PlusPlus, center: &[f32], sigma: f32) -> Vec<f32> {
    let mut v: Vec<f32> = center.iter().copied().collect();
    for x in v.iter_mut() {
        let u1: f32 = rng.gen_range(1e-7..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        *x += sigma * g;
    }
    normalize(&mut v);
    v
}

pub fn synthesize(
    n: usize,
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    sigma: f32,
    k: usize,
    seed: u64,
) -> Dataset {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut centers = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        centers.push(sample_sphere(&mut rng, dim));
    }

    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % n_clusters];
        vectors.push(perturb(&mut rng, c, sigma));
    }

    let mut q_rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(0xDEADBEEF));
    let mut queries = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        // Mix of in-cluster queries (easy) and off-center queries (hard).
        if i % 3 == 0 {
            queries.push(sample_sphere(&mut q_rng, dim));
        } else {
            let c = &centers[(i * 7) % n_clusters];
            queries.push(perturb(&mut q_rng, c, sigma * 1.5));
        }
    }

    // Ground truth via brute force.
    let mut gt = Vec::with_capacity(n_queries);
    for q in &queries {
        let mut scored: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (cos_dist(q, v), i as u32))
            .collect();
        scored.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut top: Vec<(f32, u32)> = scored.into_iter().take(k).collect();
        top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        gt.push(top.into_iter().map(|(_, i)| i).collect());
    }

    Dataset { dim, vectors, queries, gt, k }
}
