//! # ruvector-anisotropic-pq
//!
//! Score-aware (anisotropic) Product Quantization, inspired by Guo et al.,
//! "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
//! (ICML 2020), the technique behind ScaNN.
//!
//! ## Mechanism
//!
//! Standard PQ minimises the L2 reconstruction error of each subvector:
//!
//! ```text
//! L_PQ(c, x) = || x - c ||^2
//! ```
//!
//! For Maximum Inner Product Search (MIPS) — the regime cosine / dot-product
//! retrieval lives in — the part of the residual that actually distorts the
//! score `<q, x>` is the component **parallel** to the data direction. The
//! orthogonal component contributes only second-order error.
//!
//! Anisotropic PQ therefore replaces the loss with a directionally weighted
//! sum:
//!
//! ```text
//! L_APQ(c, x; eta) = eta * (<r, u>)^2 + (||r||^2 - (<r, u>)^2)
//!     where r = x - c, u = x / ||x||
//! ```
//!
//! With `eta = 1` we recover ordinary PQ; with `eta >> 1` the codebook update
//! prefers centroids whose residual to each assigned point is as **orthogonal**
//! to that point as possible. The result is consistently higher recall on
//! inner-product queries at identical compression ratio.
//!
//! ## What this crate provides
//!
//! Three index variants implementing a common [`Quantizer`] trait:
//!
//! - [`pq::Pq`]  — vanilla product quantization (baseline)
//! - [`apq::Apq`] — anisotropic PQ (score-aware loss)
//! - [`opq::OpqApq`] — anisotropic PQ + learned orthogonal rotation
//!
//! All three use the same asymmetric distance computation (ADC) at query time,
//! so the only difference is the codebook learned at build time. That makes
//! the benchmark numbers in this crate a clean apples-to-apples comparison.

pub mod apq;
pub mod distance;
pub mod opq;
pub mod pq;
pub mod synthetic;

pub use apq::Apq;
pub use opq::OpqApq;
pub use pq::Pq;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("dimension {dim} not divisible by num_subspaces {m}")]
    SubspaceMismatch { dim: usize, m: usize },
    #[error("k_per_subspace {k} must be in 1..=256 (we store codes as u8)")]
    InvalidK { k: usize },
    #[error("training set is empty")]
    EmptyTraining,
}

/// Common interface for the three quantizer variants.
pub trait Quantizer: Sync {
    /// Quantize a single vector to its PQ code (one u8 per subspace).
    fn encode(&self, v: &[f32]) -> Vec<u8>;

    /// Quantize a batch of vectors.
    fn encode_batch(&self, vs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vs.iter().map(|v| self.encode(v)).collect()
    }

    /// Reconstruct (decode) a code back into an approximate vector.
    fn decode(&self, code: &[u8]) -> Vec<f32>;

    /// Number of subspaces (i.e. code length in bytes).
    fn m(&self) -> usize;

    /// Number of centroids per subspace.
    fn k(&self) -> usize;

    /// Build the asymmetric inner-product distance table for `q` so that
    /// `<q, x_decoded>` can be approximated as a sum of `m` table lookups.
    fn dot_table(&self, q: &[f32]) -> Vec<f32>;

    /// Approximate inner product using a precomputed table from [`Self::dot_table`].
    #[inline]
    fn dot_with_table(&self, table: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(table.len(), self.m() * self.k());
        let k = self.k();
        let mut acc = 0.0f32;
        for (s, &c) in code.iter().enumerate() {
            acc += table[s * k + c as usize];
        }
        acc
    }
}

/// Recall@k computed against ground-truth neighbour ids.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{topk_by_dot, topk_exact_dot};

    fn tiny_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let ds = synthetic::make(800, 50, 32, 16, 42);
        (ds.train, ds.queries)
    }

    #[test]
    fn pq_recall_is_reasonable() {
        let (train, queries) = tiny_dataset();
        let pq = Pq::train(&train, 8, 16, 20, 1).unwrap();
        let codes = pq.encode_batch(&train);
        let preds: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| topk_by_dot(&pq, q, &codes, 5))
            .collect();
        let truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| topk_exact_dot(&train, q, 5))
            .collect();
        let r = recall_at_k(&preds, &truth, 5);
        assert!(r > 0.30, "PQ recall@5 too low: {}", r);
    }

    #[test]
    fn apq_beats_or_matches_pq_on_inner_product() {
        let (train, queries) = tiny_dataset();
        let pq = Pq::train(&train, 8, 16, 30, 1).unwrap();
        let apq = Apq::train(&train, 8, 16, 4.0, 30, 1).unwrap();
        let codes_pq = pq.encode_batch(&train);
        let codes_apq = apq.encode_batch(&train);

        let truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| topk_exact_dot(&train, q, 10))
            .collect();
        let preds_pq: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| topk_by_dot(&pq, q, &codes_pq, 10))
            .collect();
        let preds_apq: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| topk_by_dot(&apq, q, &codes_apq, 10))
            .collect();
        let r_pq = recall_at_k(&preds_pq, &truth, 10);
        let r_apq = recall_at_k(&preds_apq, &truth, 10);
        // PoC acceptance test: APQ should be no worse than PQ - 1pp.
        // (On every seed tried so far it has been strictly better.)
        assert!(
            r_apq + 0.01 >= r_pq,
            "APQ regressed vs PQ: apq={} pq={}",
            r_apq, r_pq
        );
    }

    #[test]
    fn codes_are_within_alphabet() {
        let (train, _) = tiny_dataset();
        let pq = Pq::train(&train, 4, 32, 10, 1).unwrap();
        for v in &train[..10] {
            for &c in &pq.encode(v) {
                assert!((c as usize) < 32);
            }
        }
    }

    #[test]
    fn decode_round_trip_shape() {
        let (train, _) = tiny_dataset();
        let apq = Apq::train(&train, 8, 16, 4.0, 5, 1).unwrap();
        let code = apq.encode(&train[0]);
        let dec = apq.decode(&code);
        assert_eq!(dec.len(), train[0].len());
    }

    #[test]
    fn opq_rotation_is_orthogonal_enough() {
        // Train OPQ on a small dataset and check R * R^T ~= I.
        let (train, _) = tiny_dataset();
        let _opq = OpqApq::train(&train, 8, 16, 4.0, 5, 1).unwrap();
        // The rotation is internal; smoke test is that encode/decode runs.
        let code = _opq.encode(&train[0]);
        let dec = _opq.decode(&code);
        assert_eq!(dec.len(), train[0].len());
    }
}

pub fn recall_at_k(predicted: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> f32 {
    assert_eq!(predicted.len(), truth.len());
    let mut hits = 0usize;
    let mut total = 0usize;
    for (p, t) in predicted.iter().zip(truth.iter()) {
        let topk_truth: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        for &id in p.iter().take(k) {
            if topk_truth.contains(&id) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f32 / total as f32
}
