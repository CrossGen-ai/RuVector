//! Asymmetric distance estimator.
//!
//! Given a prepared query q_rot (padded_dim, float) and a packed code c
//! for a vector x whose rotated unit form is u ~ decode(c), estimate:
//!   <q, x> = ||x|| * <q_rot, u>
//! and
//!   ||q - x||^2 = ||q||^2 + ||x||^2 - 2 * <q, x>.
//!
//! Decoding is a per-coordinate table lookup + accumulate. A B-level LUT
//! of q-side partials (one per codebook entry) could speed this up with
//! SIMD gather; we keep the naive path for clarity and still measure
//! real throughput.

use crate::quantize::{unpack_bits, ExtendedRaBitQ, Index};

#[inline]
pub fn inner_rotated(quantizer: &ExtendedRaBitQ, q_rot: &[f32], codes: &[u8]) -> f32 {
    let scale = 1.0 / (quantizer.padded_dim as f32).sqrt();
    let n = quantizer.padded_dim;
    let unpacked = unpack_bits(codes, quantizer.bits, n);
    let mut acc = 0.0f32;
    for (q, c) in q_rot.iter().zip(&unpacked) {
        acc += q * quantizer.codebook.decode(*c);
    }
    acc * scale
}

/// Estimated inner product between query and the i-th stored vector.
pub fn score(index: &Index, q_rot: &[f32], i: usize) -> f32 {
    index.norms[i] * inner_rotated(&index.quantizer, q_rot, &index.codes[i])
}

/// Estimated L2^2 distance between query and the i-th stored vector.
pub fn l2_sq(index: &Index, q_rot: &[f32], q_norm_sq: f32, i: usize) -> f32 {
    let ip = score(index, q_rot, i);
    let xn = index.norms[i];
    q_norm_sq + xn * xn - 2.0 * ip
}

/// Brute scan returning top-k by estimated L2 distance (ascending).
pub fn topk_l2(index: &Index, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    assert!(!index.is_empty());
    let q_rot = index.quantizer.prepare_query(query).expect("dim mismatch");
    let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    let mut scored: Vec<(usize, f32)> = (0..index.len())
        .map(|i| (i, l2_sq(index, &q_rot, q_norm_sq, i)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

/// Brute scan returning top-k by estimated inner product (descending).
pub fn topk_ip(index: &Index, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    assert!(!index.is_empty());
    let q_rot = index.quantizer.prepare_query(query).expect("dim mismatch");
    let mut scored: Vec<(usize, f32)> = (0..index.len())
        .map(|i| (i, score(index, &q_rot, i)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::ExtendedRaBitQ;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn estimator_tracks_exact_ip_within_bound() {
        let dim = 128;
        let q = ExtendedRaBitQ::new(dim, 4, 42).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let n = 200;
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let mut idx = Index::new(q);
        idx.add(&xs).unwrap();
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let q_rot = idx.quantizer.prepare_query(&query).unwrap();
        let mut rel_errs = vec![];
        for i in 0..n {
            let est = score(&idx, &q_rot, i);
            let exact: f32 = query.iter().zip(&xs[i]).map(|(a, b)| a * b).sum();
            if exact.abs() > 1e-3 {
                rel_errs.push(((est - exact) / exact).abs());
            }
        }
        rel_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = rel_errs[rel_errs.len() / 2];
        // 4-bit should give median relative IP error well under 10%.
        assert!(median < 0.10, "median rel err {} too high", median);
    }
}
