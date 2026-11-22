//! Swappable short-code reranker backends.
//!
//! Each backend encodes a vector into a compact code and provides a ranking
//! distance over codes. `distance` need not be metric — only order-preserving
//! enough to shortlist for a final full-precision rerank.

use crate::{gaussian_matrix, l2sq, pack_bins, popcount_xor, ScalarRange};

/// Any shortlist/rerank backend.
pub trait Reranker: Send + Sync {
    fn encode(&self, v: &[f32]) -> Vec<u8>;
    fn code_bytes(&self) -> usize;
    /// Smaller = better (farther from `a` is larger).
    fn distance(&self, a: &[u8], b: &[u8]) -> u32;
    fn name(&self) -> &'static str;
}

/// Gray Code SQ. Bits-per-dim in {1,2,4,8}; Hamming distance ranks.
pub struct GcsqReranker {
    pub range: ScalarRange,
    pub code_bytes: usize,
}

impl GcsqReranker {
    pub fn fit(train: &[Vec<f32>], bits_per_dim: u8) -> Self {
        let range = ScalarRange::fit(train, bits_per_dim);
        let total_bits = range.dim * bits_per_dim as usize;
        let code_bytes = (total_bits + 7) / 8;
        Self { range, code_bytes }
    }
}

impl Reranker for GcsqReranker {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let bins = self.range.bins(v);
        pack_bins(&bins, self.range.bits_per_dim, true)
    }
    fn code_bytes(&self) -> usize {
        self.code_bytes
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> u32 {
        popcount_xor(a, b)
    }
    fn name(&self) -> &'static str {
        "gcsq"
    }
}

/// Plain binary (non-Gray) SQ at the same bitrate; Hamming ranks.
pub struct BinarySqReranker {
    pub range: ScalarRange,
    pub code_bytes: usize,
}

impl BinarySqReranker {
    pub fn fit(train: &[Vec<f32>], bits_per_dim: u8) -> Self {
        let range = ScalarRange::fit(train, bits_per_dim);
        let total_bits = range.dim * bits_per_dim as usize;
        let code_bytes = (total_bits + 7) / 8;
        Self { range, code_bytes }
    }
}

impl Reranker for BinarySqReranker {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let bins = self.range.bins(v);
        pack_bins(&bins, self.range.bits_per_dim, false)
    }
    fn code_bytes(&self) -> usize {
        self.code_bytes
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> u32 {
        popcount_xor(a, b)
    }
    fn name(&self) -> &'static str {
        "binary_sq"
    }
}

/// SimHash baseline: sign(R x) where R is a `n_bits × dim` Gaussian.
/// Code width is matched to GCSQ by making `n_bits = dim * bits_per_dim`.
pub struct SimHashReranker {
    pub dim: usize,
    pub n_bits: usize,
    pub rmat: Vec<f32>, // row-major, n_bits x dim
    pub code_bytes: usize,
}

impl SimHashReranker {
    pub fn fit(dim: usize, n_bits: usize, seed: u64) -> Self {
        let rmat = gaussian_matrix(seed, n_bits, dim);
        let code_bytes = (n_bits + 7) / 8;
        Self {
            dim,
            n_bits,
            rmat,
            code_bytes,
        }
    }
}

impl Reranker for SimHashReranker {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim);
        let mut out = vec![0u8; self.code_bytes];
        for r in 0..self.n_bits {
            let row = &self.rmat[r * self.dim..(r + 1) * self.dim];
            let mut dot = 0.0f32;
            for (a, b) in row.iter().zip(v.iter()) {
                dot += a * b;
            }
            if dot >= 0.0 {
                out[r / 8] |= 1 << (r % 8);
            }
        }
        out
    }
    fn code_bytes(&self) -> usize {
        self.code_bytes
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> u32 {
        popcount_xor(a, b)
    }
    fn name(&self) -> &'static str {
        "simhash"
    }
}

/// Quality harness: measure how well a reranker's code distance correlates
/// with full-precision L2 distance on a probe set.
///
/// Returns (spearman_rho, recall_at_k) where spearman is computed over all
/// (query, base) pairs, and recall@k is the mean over queries of
/// |topk_code ∩ topk_l2| / k.
pub fn evaluate(
    backend: &dyn Reranker,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> EvalResult {
    let base_codes: Vec<Vec<u8>> = base.iter().map(|v| backend.encode(v)).collect();
    let query_codes: Vec<Vec<u8>> = queries.iter().map(|v| backend.encode(v)).collect();

    let n = base.len();
    let mut all_code_d: Vec<f64> = Vec::with_capacity(queries.len() * n);
    let mut all_true_d: Vec<f64> = Vec::with_capacity(queries.len() * n);
    let mut recall_sum = 0.0f64;

    for (qi, q) in queries.iter().enumerate() {
        let qc = &query_codes[qi];
        let mut pairs: Vec<(usize, u32, f32)> = (0..n)
            .map(|i| (i, backend.distance(qc, &base_codes[i]), l2sq(q, &base[i])))
            .collect();
        for (_, cd, td) in &pairs {
            all_code_d.push(*cd as f64);
            all_true_d.push(*td as f64);
        }
        // Top-k by true L2
        let mut true_order = pairs.clone();
        true_order.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let topk_true: std::collections::HashSet<usize> =
            true_order.iter().take(k).map(|p| p.0).collect();
        // Top-k by code
        pairs.sort_by_key(|p| p.1);
        let topk_code: std::collections::HashSet<usize> =
            pairs.iter().take(k).map(|p| p.0).collect();
        let overlap = topk_true.intersection(&topk_code).count();
        recall_sum += overlap as f64 / k as f64;
    }
    let rho = spearman(&all_code_d, &all_true_d);
    EvalResult {
        name: backend.name(),
        code_bytes: backend.code_bytes(),
        spearman_rho: rho,
        recall_at_k: recall_sum / queries.len() as f64,
        k,
    }
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub name: &'static str,
    pub code_bytes: usize,
    pub spearman_rho: f64,
    pub recall_at_k: f64,
    pub k: usize,
}

/// Spearman rank correlation. Ties are broken by original order (sufficient
/// for monotonic-proxy evaluation; avoids ties-adjustment for simplicity).
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let rx = ranks(x);
    let ry = ranks(y);
    let mean = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let a = rx[i] - mean;
        let b = ry[i] - mean;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom == 0.0 { 0.0 } else { num / denom }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
    let mut r = vec![0.0f64; v.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = rank as f64;
    }
    r
}
