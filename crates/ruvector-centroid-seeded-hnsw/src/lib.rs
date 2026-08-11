//! Centroid-seeded entry points for greedy k-NN graph search.
//!
//! HNSW's classical entry point is a single fixed node at the top layer,
//! forcing a long greedy descent for out-of-distribution queries. This crate
//! studies **seeder strategies** — how to choose entry points — while holding
//! the underlying k-NN graph constant. We use a flat k-NN graph as a
//! reproducible proxy for HNSW's upper-layer greedy search.
//!
//! See `README.md` in `docs/research/nightly/2026-08-10-centroid-seeded-hnsw/`
//! for the full write-up.

pub mod graph;
pub mod kmeans;
pub mod seed;

#[cfg(test)]
mod tests;

pub use graph::{KnnGraph, SearchStats};
pub use seed::{CentroidSeeder, MultiCentroidSeeder, RandomSeeder, Seeder};

/// Squared Euclidean distance.
#[inline]
pub fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic tiny LCG for reproducibility (no `rand` dep).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound.max(1)
    }
    pub fn gauss(&mut self) -> f32 {
        // Box-Muller
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() as f32 * (2.0 * std::f64::consts::PI * u2).cos() as f32
    }
}

/// Generate a synthetic multi-cluster dataset (N points, D dims, K clusters).
pub fn gen_clusters(n: usize, d: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    // Cluster centers scattered on a large sphere.
    let mut centers = Vec::with_capacity(k);
    for _ in 0..k {
        let mut c = vec![0f32; d];
        for x in &mut c {
            *x = rng.gauss() * 10.0;
        }
        centers.push(c);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % k];
        let mut v = vec![0f32; d];
        for j in 0..d {
            v[j] = c[j] + rng.gauss();
        }
        out.push(v);
    }
    out
}

/// Exact top-1 (for recall ground truth).
pub fn brute_top1(data: &[Vec<f32>], q: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, v) in data.iter().enumerate() {
        let d = sqdist(v, q);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}
