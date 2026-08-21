//! Distance oracles: swappable backends behind a single trait.

use crate::quantize::{
    dequant_int8, dim_means, hamming_u64, quantize_binary, quantize_int8,
};

/// A distance oracle answers "what is the distance between item `i` and the
/// query I was primed with?" Oracles are the swappable point of the cascade.
pub trait DistanceOracle: Send + Sync {
    /// Number of stored items.
    fn len(&self) -> usize;
    /// True if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Prime the oracle with a new query. Called once per search.
    fn prime(&mut self, query: &[f32]);
    /// Score item `i` against the primed query. Lower is closer.
    fn score(&self, i: usize) -> f32;
    /// Approximate memory footprint in bytes.
    fn footprint_bytes(&self) -> usize;
    /// Human-facing oracle name for reports.
    fn name(&self) -> &'static str;
}

// -------- FP32 exact --------

/// Exact FP32 L2 oracle (baseline). One `dim`-sized read per score call.
pub struct Fp32Oracle {
    dim: usize,
    data: Vec<f32>,
    query: Vec<f32>,
}

impl Fp32Oracle {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            assert_eq!(v.len(), dim, "ragged input");
            data.extend_from_slice(v);
        }
        Self { dim, data, query: vec![0.0; dim] }
    }
}

impl DistanceOracle for Fp32Oracle {
    fn len(&self) -> usize {
        if self.dim == 0 { 0 } else { self.data.len() / self.dim }
    }
    fn prime(&mut self, query: &[f32]) {
        assert_eq!(query.len(), self.dim);
        self.query.copy_from_slice(query);
    }
    fn score(&self, i: usize) -> f32 {
        let base = i * self.dim;
        let row = &self.data[base..base + self.dim];
        let mut acc = 0.0f32;
        for j in 0..self.dim {
            let d = row[j] - self.query[j];
            acc += d * d;
        }
        acc
    }
    fn footprint_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
    fn name(&self) -> &'static str { "fp32" }
}

// -------- INT8 per-vector affine --------

/// Per-vector affine INT8 oracle. Uses (lo, hi) window per vector, which
/// avoids the "one outlier tanks the whole codebook" failure of a single
/// global window. Score is L2 on dequantized floats — this is deliberately
/// simple; the point is to compare bandwidth, not to chase peak accuracy.
pub struct Int8Oracle {
    dim: usize,
    bytes: Vec<u8>,       // len * dim
    windows: Vec<(f32, f32)>,
    query: Vec<f32>,
}

impl Int8Oracle {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut bytes = Vec::with_capacity(vectors.len() * dim);
        let mut windows = Vec::with_capacity(vectors.len());
        for v in vectors {
            assert_eq!(v.len(), dim, "ragged input");
            let (b, lo, hi) = quantize_int8(v);
            bytes.extend_from_slice(&b);
            windows.push((lo, hi));
        }
        Self { dim, bytes, windows, query: vec![0.0; dim] }
    }
}

impl DistanceOracle for Int8Oracle {
    fn len(&self) -> usize { self.windows.len() }
    fn prime(&mut self, query: &[f32]) {
        assert_eq!(query.len(), self.dim);
        self.query.copy_from_slice(query);
    }
    fn score(&self, i: usize) -> f32 {
        let base = i * self.dim;
        let row = &self.bytes[base..base + self.dim];
        let (lo, hi) = self.windows[i];
        let mut acc = 0.0f32;
        for j in 0..self.dim {
            let d = dequant_int8(row[j], lo, hi) - self.query[j];
            acc += d * d;
        }
        acc
    }
    fn footprint_bytes(&self) -> usize {
        self.bytes.len() + self.windows.len() * 8
    }
    fn name(&self) -> &'static str { "int8" }
}

// -------- 1-bit Hamming --------

/// 1-bit sign quantization with a per-dimension mean threshold learned
/// from the training set. Score is Hamming distance (POPCNT).
pub struct HammingOracle {
    dim: usize,
    words_per_vec: usize,
    codes: Vec<u64>,
    thresholds: Vec<f32>,
    query_code: Vec<u64>,
}

impl HammingOracle {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let thresholds = dim_means(vectors, dim);
        let words_per_vec = dim.div_ceil(64);
        let mut codes = Vec::with_capacity(vectors.len() * words_per_vec);
        for v in vectors {
            assert_eq!(v.len(), dim, "ragged input");
            codes.extend_from_slice(&quantize_binary(v, &thresholds));
        }
        Self {
            dim,
            words_per_vec,
            codes,
            thresholds,
            query_code: vec![0u64; words_per_vec],
        }
    }
}

impl DistanceOracle for HammingOracle {
    fn len(&self) -> usize {
        if self.words_per_vec == 0 { 0 } else { self.codes.len() / self.words_per_vec }
    }
    fn prime(&mut self, query: &[f32]) {
        assert_eq!(query.len(), self.dim);
        let q = quantize_binary(query, &self.thresholds);
        self.query_code.copy_from_slice(&q);
    }
    fn score(&self, i: usize) -> f32 {
        let base = i * self.words_per_vec;
        let row = &self.codes[base..base + self.words_per_vec];
        hamming_u64(row, &self.query_code) as f32
    }
    fn footprint_bytes(&self) -> usize {
        self.codes.len() * 8 + self.thresholds.len() * 4
    }
    fn name(&self) -> &'static str { "hamming" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn fp32_finds_exact() {
        let mut o = Fp32Oracle::from_vectors(&toy());
        o.prime(&[1.0, 0.0, 0.0, 0.0]);
        assert!(o.score(0) < 1e-6);
        assert!(o.score(1) > 0.5);
    }

    #[test]
    fn int8_close_to_fp32() {
        let vs = toy();
        let mut a = Fp32Oracle::from_vectors(&vs);
        let mut b = Int8Oracle::from_vectors(&vs);
        let q = vec![0.9, 0.0, 0.0, 0.1];
        a.prime(&q); b.prime(&q);
        for i in 0..vs.len() {
            let da = a.score(i);
            let db = b.score(i);
            assert!((da - db).abs() < 0.2, "int8 drift {da} vs {db} at {i}");
        }
    }

    #[test]
    fn hamming_zero_for_self() {
        let vs = toy();
        let mut h = HammingOracle::from_vectors(&vs);
        h.prime(&vs[0]);
        assert!(h.score(0) < 0.5);
    }

    #[test]
    fn footprint_shrinks_across_oracles() {
        let vs: Vec<Vec<f32>> = (0..64)
            .map(|i| (0..128).map(|j| ((i + j) as f32).sin()).collect())
            .collect();
        let f = Fp32Oracle::from_vectors(&vs).footprint_bytes();
        let i8 = Int8Oracle::from_vectors(&vs).footprint_bytes();
        let hm = HammingOracle::from_vectors(&vs).footprint_bytes();
        assert!(i8 < f);
        assert!(hm < i8);
    }
}
