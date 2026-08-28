//! ruvector-finger — Fast INference for Graph-based approximate NEaR-neighbor search.
//!
//! This crate implements a self-contained, HNSW-agnostic version of the FINGER
//! distance-approximation trick (Yin et al., WWW 2023) plus a Johnson–Lindenstrauss
//! (JL) baseline for comparison. Both estimators share the same
//! [`DistanceEstimator`] trait so they can be swapped inside a real HNSW/ACORN
//! graph search loop.
//!
//! # Layout
//! * [`Dataset`]       – deterministic synthetic dataset (unit-normalised f32 vectors).
//! * [`PivotIndex`]    – flat pivot-based structure that partitions vectors around
//!   anchors; supplies the "arrival pivot" for FINGER-style approximation.
//! * [`DistanceEstimator`] – trait giving a per-pivot handle used during search.
//! * [`ExactEstimator`], [`JlEstimator`], [`FingerEstimator`] – three concrete
//!   variants required by the ADR (baseline + 2 alternatives).
//! * [`recall_at_k`]    – utility for measuring recall against brute-force truth.
//!
//! The implementation is intentionally dependency-light (rand + rand_distr only)
//! so it can be reasoned about end-to-end in a single readable file.

pub mod estimator;
pub mod exact;
pub mod finger;
pub mod jl;
pub mod pivot;

pub use estimator::{DistanceEstimator, PivotHandle, SearchStats};
pub use exact::ExactEstimator;
pub use finger::FingerEstimator;
pub use jl::JlEstimator;
pub use pivot::PivotIndex;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// A deterministic synthetic dataset of unit-norm f32 vectors drawn from a
/// low-rank Gaussian mixture. Real numbers reflect a realistic embedding
/// distribution (dense, low intrinsic dimension, unit norm) without pulling in
/// heavy dependencies. Seeded so nightly benchmarks are reproducible.
#[derive(Clone)]
pub struct Dataset {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn synthetic(n_vectors: usize, n_queries: usize, dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // 32 latent factors → typical of instruction-tuned sentence embeddings.
        let n_latent = 32.min(dim);
        let factors: Vec<Vec<f32>> = (0..n_latent)
            .map(|_| gaussian_unit(&mut rng, dim))
            .collect();

        let sample = |rng: &mut StdRng| -> Vec<f32> {
            let mut v = vec![0f32; dim];
            for f in &factors {
                let w: f32 = rng.sample::<f32, _>(StandardNormal);
                for i in 0..dim {
                    v[i] += w * f[i];
                }
            }
            // add mild isotropic noise so intrinsic dim < d.
            for x in v.iter_mut() {
                let n: f32 = rng.sample::<f32, _>(StandardNormal);
                *x += 0.1 * n;
            }
            normalise(&mut v);
            v
        };

        let vectors: Vec<Vec<f32>> = (0..n_vectors).map(|_| sample(&mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| sample(&mut rng)).collect();
        Self { dim, vectors, queries }
    }

    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }
}

/// Draw a unit-norm Gaussian vector.
pub(crate) fn gaussian_unit(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    normalise(&mut v);
    v
}

pub(crate) fn normalise(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Brute-force ground truth: highest inner-product top-k neighbours.
pub fn brute_force_topk(dataset: &Dataset, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = dataset
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (dot(query, v), i as u32))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Recall@k between predicted ids and ground truth ids for a single query.
pub fn recall_at_k(pred: &[u32], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for t in truth {
        if pred.contains(t) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_vectors_are_unit_norm() {
        let ds = Dataset::synthetic(64, 4, 32, 7);
        for v in &ds.vectors {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "norm={n}");
        }
    }

    #[test]
    fn brute_force_is_deterministic() {
        let ds = Dataset::synthetic(200, 5, 48, 42);
        let a = brute_force_topk(&ds, &ds.queries[0], 10);
        let b = brute_force_topk(&ds, &ds.queries[0], 10);
        assert_eq!(a, b);
    }
}
