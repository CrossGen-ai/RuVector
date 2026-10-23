//! Distance helpers, ground-truth brute force, and dataset generators.

use crate::quant::Codec;

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Exact brute-force top-k by L2 distance. Returns the ids in ascending-distance order.
pub fn brute_force_topk(query: &[f32], data: &[f32], dim: usize, k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for i in 0..n {
        let d = l2_sq(query, &data[i * dim..(i + 1) * dim]);
        heap.push((d, i as u32));
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap.truncate(k);
    heap.into_iter().map(|(_, id)| id).collect()
}

/// Top-k over a codec, scanning every encoded vector asymmetrically.
pub fn codec_topk<C: Codec>(
    codec: &C,
    query: &[f32],
    codes: &[u8],
    n: usize,
    k: usize,
) -> Vec<u32> {
    let stride = codec.code_bytes();
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let d = codec.l2_sq_asymmetric(query, &codes[i * stride..(i + 1) * stride]);
        heap.push((d, i as u32));
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap.truncate(k);
    heap.into_iter().map(|(_, id)| id).collect()
}

/// Recall of `got` against `truth` (both top-k id lists).
pub fn recall(got: &[u32], truth: &[u32]) -> f32 {
    let mut hits = 0usize;
    for g in got {
        if truth.contains(g) { hits += 1; }
    }
    hits as f32 / truth.len() as f32
}

/// Average recall@k over `queries` using a codec, with raw ground-truth from `data`.
pub fn brute_force_recall<C: Codec>(
    codec: &C,
    data: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> f32 {
    let n = data.len() / dim;
    let nq = queries.len() / dim;
    // Encode the base.
    let stride = codec.code_bytes();
    let mut codes = vec![0u8; n * stride];
    for i in 0..n {
        codec.encode(
            &data[i * dim..(i + 1) * dim],
            &mut codes[i * stride..(i + 1) * stride],
        );
    }
    let mut total = 0.0f32;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let truth = brute_force_topk(query, data, dim, k);
        let got = codec_topk(codec, query, &codes, n, k);
        total += recall(&got, &truth);
    }
    total / nq as f32
}

// -------- Dataset generators --------

/// Deterministic LCG uniform in [-1, 1].
fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = ((*state >> 33) as u32) as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}

/// Mimic a real-world embedding failure mode: typical entries are in [-1, 1],
/// but a small fraction of vectors have a *sparse outlier spike* on one or
/// two dimensions (e.g. value ±25). This is exactly what happens with text
/// embeddings (one or two "topic" dims occasionally dominate a vector) and
/// what defeats naive global-range SQ: rare outliers inflate the min/max
/// envelope and crush the resolution of every other entry. Randomised
/// Hadamard rotation smears each outlier across all output dimensions,
/// shrinking the global envelope and restoring resolution.
pub fn make_skewed_dataset(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut data = vec![0.0f32; n * d];
    let mut state = seed;
    for i in 0..n {
        for j in 0..d {
            data[i * d + j] = lcg(&mut state);
        }
        // ~15 % of vectors carry one outlier spike on a random dim.
        let spike_prob = (lcg(&mut state) + 1.0) * 0.5; // uniform in [0, 1]
        if spike_prob < 0.15 {
            let idx = (((lcg(&mut state) + 1.0) * 0.5) * d as f32) as usize;
            let idx = idx.min(d - 1);
            let sign = if lcg(&mut state) > 0.0 { 1.0 } else { -1.0 };
            data[i * d + idx] = sign * 25.0;
        }
    }
    data
}

/// Standard Gaussian-ish uniform dataset (no skew). Hadamard should at least
/// tie here — this measures the "no-regression" condition.
pub fn make_uniform_dataset(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut data = vec![0.0f32; n * d];
    let mut state = seed;
    for v in &mut data { *v = lcg(&mut state); }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{HadamardSQ4, UniformSQ4};

    #[test]
    fn hadamard_sq4_beats_uniform_recall_on_skew() {
        let d = 64;
        let n = 2000;
        let nq = 32;
        let k = 10;
        let data = make_skewed_dataset(n, d, 42);
        let queries = make_skewed_dataset(nq, d, 1337);
        let base = UniformSQ4::train(d, &data);
        let hada = HadamardSQ4::train(d, &data, 7);
        let r_base = brute_force_recall(&base, &data, &queries, d, k);
        let r_hada = brute_force_recall(&hada, &data, &queries, d, k);
        assert!(
            r_hada > r_base + 0.05,
            "expected Hadamard SQ4 to beat uniform by >=0.05: base={} hada={}",
            r_base,
            r_hada
        );
    }

    #[test]
    fn recall_bounds_sanity() {
        let d = 16;
        let n = 128;
        let k = 10;
        let data = make_uniform_dataset(n, d, 1);
        let queries = make_uniform_dataset(8, d, 2);
        let hada = HadamardSQ4::train(d, &data, 3);
        let r = brute_force_recall(&hada, &data, &queries, d, k);
        assert!((0.0..=1.0).contains(&r));
    }
}
