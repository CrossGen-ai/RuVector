//! # ruvector-anisotropic-pq
//!
//! Anisotropic Product Quantization for Maximum Inner Product Search (MIPS).
//!
//! Standard Product Quantization (PQ) trains sub-codebooks by minimizing L2
//! reconstruction error `||x - x̂||²`. That is optimal for L2 nearest-neighbor
//! search but wastes bits for MIPS: the score error `<q, x> - <q, x̂>` depends
//! only on the component of `x - x̂` that is *parallel* to `q`. The orthogonal
//! component contributes nothing to the score.
//!
//! Following the ScaNN paper (Guo et al., ICML 2020,
//! "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"),
//! this crate implements a weighted quantization loss:
//!
//! ```text
//! L(x, c) = h_par · (r · û)² + h_orth · (||r||² - (r · û)²)
//!         = h_orth · ||r||² + (h_par - h_orth) · (r · û)²
//!   where r = x - c,  û = x / ||x||,  and eta = h_par / h_orth ≥ 1
//! ```
//!
//! Setting `eta = 1` recovers standard k-means. Setting `eta >> 1` forces
//! centroids to reconstruct the parallel (score-relevant) component precisely,
//! trading off orthogonal fidelity that MIPS does not need.
//!
//! Three variants are exposed via the [`Pq`] trait:
//!
//! * `StandardPq` — plain L2-optimized k-means codebook (eta = 1.0).
//! * `AnisotropicPq { eta: 4.0 }`  — moderate parallel weighting.
//! * `AnisotropicPq { eta: 16.0 }` — heavy parallel weighting (ScaNN-like).
//!
//! At query time a Look-Up Table (LUT) is precomputed once per query per
//! subspace, then the score is `sum_m LUT[m][code_m]` per database vector.

#![warn(missing_debug_implementations)]

pub mod dataset;
pub mod math;
pub mod standard_pq;
pub mod anisotropic_pq;

/// A single scored database hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

/// Product-quantization configuration.
#[derive(Debug, Clone, Copy)]
pub struct PqConfig {
    /// Total vector dimension.
    pub dim: usize,
    /// Number of subspaces. Must divide `dim`.
    pub m: usize,
    /// Number of centroids per subspace. Fixed to 256 (u8 codes).
    pub k: usize,
    /// k-means iterations per subspace.
    pub iterations: usize,
    /// Reproducible RNG seed.
    pub seed: u64,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self { dim: 128, m: 8, k: 256, iterations: 12, seed: 0xA1_5E_D5_00 }
    }
}

/// Trait for MIPS-oriented quantizers.
pub trait Pq: Send + Sync + std::fmt::Debug {
    /// Human-readable variant name.
    fn name(&self) -> &str;

    /// Train the codebook on `data` (`n × dim` row-major).
    fn train(&mut self, data: &[f32], n: usize);

    /// Encode `n` vectors into `n · m` byte codes.
    fn encode(&self, data: &[f32], n: usize) -> Vec<u8>;

    /// Top-`k` MIPS results over encoded database.
    fn search(&self, query: &[f32], codes: &[u8], n: usize, k: usize) -> Vec<Hit>;

    /// Mean reconstruction error on validation set (Σ ||x - x̂||²).
    fn reconstruction_error(&self, data: &[f32], n: usize) -> f64;

    /// Approximate memory footprint of the codebook, in bytes.
    fn memory_bytes(&self) -> usize;
}

/// Deterministic LCG for reproducible dataset + init without external deps.
#[derive(Debug, Clone, Copy)]
pub struct Lcg(pub u64);
impl Lcg {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // Numerical Recipes constants.
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Box–Muller standard normal.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        mag * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Ground-truth exact-MIPS top-k (for recall evaluation only).
pub fn exact_mips(query: &[f32], data: &[f32], n: usize, k: usize) -> Vec<Hit> {
    let d = query.len();
    let mut scored: Vec<Hit> = (0..n)
        .map(|i| {
            let row = &data[i * d..(i + 1) * d];
            let mut s = 0.0f32;
            for j in 0..d {
                s += query[j] * row[j];
            }
            Hit { id: i as u32, score: s }
        })
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Recall@k of `predicted` vs `truth` (both top-k sets by id).
pub fn recall_at_k(predicted: &[Hit], truth: &[Hit]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let tset: std::collections::HashSet<u32> = truth.iter().map(|h| h.id).collect();
    let mut hits = 0usize;
    for p in predicted {
        if tset.contains(&p.id) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

/// Mean squared error of predicted scores vs true inner-product scores
/// on the union of predicted+truth ids. Lower means better MIPS-score fidelity.
pub fn score_mse(query: &[f32], data: &[f32], predicted: &[Hit]) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }
    let d = query.len();
    let mut acc = 0.0f64;
    for h in predicted {
        let row = &data[(h.id as usize) * d..(h.id as usize + 1) * d];
        let mut true_s = 0.0f32;
        for j in 0..d {
            true_s += query[j] * row[j];
        }
        let err = (h.score - true_s) as f64;
        acc += err * err;
    }
    acc / predicted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_reproducible() {
        let mut a = Lcg(42);
        let mut b = Lcg(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn exact_mips_orders_correctly() {
        let data = vec![
            1.0, 0.0, // id 0, score 1
            0.5, 0.0, // id 1, score 0.5
            2.0, 0.0, // id 2, score 2
        ];
        let q = vec![1.0, 0.0];
        let top = exact_mips(&q, &data, 3, 3);
        assert_eq!(top[0].id, 2);
        assert_eq!(top[1].id, 0);
        assert_eq!(top[2].id, 1);
    }

    #[test]
    fn recall_perfect_and_partial() {
        let truth = vec![
            Hit { id: 1, score: 1.0 },
            Hit { id: 2, score: 0.9 },
            Hit { id: 3, score: 0.8 },
        ];
        let pred_perfect = truth.clone();
        assert!((recall_at_k(&pred_perfect, &truth) - 1.0).abs() < 1e-6);
        let pred_half = vec![
            Hit { id: 1, score: 1.0 },
            Hit { id: 99, score: 0.9 },
            Hit { id: 3, score: 0.8 },
        ];
        assert!((recall_at_k(&pred_half, &truth) - 2.0 / 3.0).abs() < 1e-6);
    }
}
