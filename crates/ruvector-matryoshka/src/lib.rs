//! Matryoshka Adaptive Retrieval (MAR)
//!
//! Coarse-to-fine ANN search built on top of Matryoshka representations
//! (Kusupati et al., NeurIPS 2022). The technique exploits the fact that
//! MRL-trained embeddings remain meaningful when truncated to any prefix
//! dimension, so we can run a cheap brute-force pass at a small prefix
//! dimension, then re-rank a short candidate list using the full vector.
//!
//! This crate is intentionally self-contained: no external graph or
//! quantizer dependency. The goal is a clean reference implementation
//! producing real numbers we can compare against full-dimension brute
//! force and prefix-only brute force.

use std::fmt;
use thiserror::Error;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[derive(Debug, Error)]
pub enum MarError {
    #[error("vector dimension {got} does not match index dimension {expected}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("low_dim {low} must be <= full_dim {full} and > 0")]
    BadLowDim { low: usize, full: usize },
    #[error("empty corpus")]
    EmptyCorpus,
}

/// A retrieval backend over a fixed corpus of L2-normalized vectors.
pub trait Retriever {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, MarError>;
    fn name(&self) -> &'static str;
    /// Total bytes held by the resident index (vectors only, no metadata).
    fn resident_bytes(&self) -> usize;
}

/// L2-normalize in place. No-op for zero vectors.
pub fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Inner product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn top_k(scores: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let mut s = scores;
    let k = k.min(s.len());
    if k == 0 {
        s.clear();
        return s;
    }
    s.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    s.truncate(k);
    s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    s
}

/// Brute-force cosine search at the full embedding dimension.
pub struct BruteForceFull {
    vectors: Vec<f32>, // row-major, n * full_dim
    full_dim: usize,
    n: usize,
}

impl BruteForceFull {
    pub fn new(vectors: Vec<f32>, full_dim: usize) -> Result<Self, MarError> {
        if vectors.is_empty() {
            return Err(MarError::EmptyCorpus);
        }
        let n = vectors.len() / full_dim;
        Ok(Self { vectors, full_dim, n })
    }
    pub fn len(&self) -> usize { self.n }
}

impl Retriever for BruteForceFull {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, MarError> {
        if query.len() != self.full_dim {
            return Err(MarError::DimensionMismatch { expected: self.full_dim, got: query.len() });
        }
        let scores: Vec<(u32, f32)> = (0..self.n)
            .map(|i| {
                let s = dot(query, &self.vectors[i * self.full_dim..(i + 1) * self.full_dim]);
                (i as u32, s)
            })
            .collect();
        Ok(top_k(scores, k))
    }
    fn name(&self) -> &'static str { "brute-full" }
    fn resident_bytes(&self) -> usize { self.vectors.len() * 4 }
}

/// Brute-force cosine search using only the first `low_dim` coordinates.
/// Each row is pre-normalized so the prefix is itself an L2-normalized vector.
pub struct BruteForceLow {
    prefixes: Vec<f32>, // n * low_dim, each row L2-normalized
    low_dim: usize,
    n: usize,
}

impl BruteForceLow {
    pub fn new(full_vectors: &[f32], full_dim: usize, low_dim: usize) -> Result<Self, MarError> {
        if low_dim == 0 || low_dim > full_dim {
            return Err(MarError::BadLowDim { low: low_dim, full: full_dim });
        }
        if full_vectors.is_empty() {
            return Err(MarError::EmptyCorpus);
        }
        let n = full_vectors.len() / full_dim;
        let mut prefixes = Vec::with_capacity(n * low_dim);
        for i in 0..n {
            let mut row: Vec<f32> = full_vectors[i * full_dim..i * full_dim + low_dim].to_vec();
            l2_normalize(&mut row);
            prefixes.extend_from_slice(&row);
        }
        Ok(Self { prefixes, low_dim, n })
    }
}

impl Retriever for BruteForceLow {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, MarError> {
        if query.len() != self.low_dim {
            return Err(MarError::DimensionMismatch { expected: self.low_dim, got: query.len() });
        }
        let scores: Vec<(u32, f32)> = (0..self.n)
            .map(|i| {
                let s = dot(query, &self.prefixes[i * self.low_dim..(i + 1) * self.low_dim]);
                (i as u32, s)
            })
            .collect();
        Ok(top_k(scores, k))
    }
    fn name(&self) -> &'static str { "brute-low" }
    fn resident_bytes(&self) -> usize { self.prefixes.len() * 4 }
}

/// Matryoshka Adaptive Retrieval.
///
/// Phase 1: brute force at `low_dim` over normalized prefixes to gather
/// `k * rerank_factor` candidates.
/// Phase 2: rescore those candidates with the full-dimension inner product
/// and return the top `k`.
pub struct MatryoshkaAdaptive {
    full: Vec<f32>,        // n * full_dim
    prefixes: Vec<f32>,    // n * low_dim, prefix-normalized
    full_dim: usize,
    low_dim: usize,
    n: usize,
    rerank_factor: usize,
}

impl MatryoshkaAdaptive {
    pub fn new(
        full_vectors: Vec<f32>,
        full_dim: usize,
        low_dim: usize,
        rerank_factor: usize,
    ) -> Result<Self, MarError> {
        if low_dim == 0 || low_dim > full_dim {
            return Err(MarError::BadLowDim { low: low_dim, full: full_dim });
        }
        if full_vectors.is_empty() {
            return Err(MarError::EmptyCorpus);
        }
        let n = full_vectors.len() / full_dim;
        let mut prefixes = Vec::with_capacity(n * low_dim);
        for i in 0..n {
            let mut row: Vec<f32> = full_vectors[i * full_dim..i * full_dim + low_dim].to_vec();
            l2_normalize(&mut row);
            prefixes.extend_from_slice(&row);
        }
        Ok(Self {
            full: full_vectors,
            prefixes,
            full_dim,
            low_dim,
            n,
            rerank_factor: rerank_factor.max(1),
        })
    }

    pub fn rerank_factor(&self) -> usize { self.rerank_factor }
    pub fn low_dim(&self) -> usize { self.low_dim }
    pub fn full_dim(&self) -> usize { self.full_dim }

    /// Search using a full-dimension query. The first `low_dim` coords of
    /// the query are re-normalized internally for the coarse pass.
    pub fn search_full_query(
        &self,
        query_full: &[f32],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, MarError> {
        if query_full.len() != self.full_dim {
            return Err(MarError::DimensionMismatch { expected: self.full_dim, got: query_full.len() });
        }
        let mut q_low: Vec<f32> = query_full[..self.low_dim].to_vec();
        l2_normalize(&mut q_low);

        let coarse_k = (k * self.rerank_factor).min(self.n).max(1);
        let coarse_scores: Vec<(u32, f32)> = (0..self.n)
            .map(|i| {
                let s = dot(&q_low, &self.prefixes[i * self.low_dim..(i + 1) * self.low_dim]);
                (i as u32, s)
            })
            .collect();
        let candidates = top_k(coarse_scores, coarse_k);

        let rescored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, _)| {
                let off = id as usize * self.full_dim;
                let s = dot(query_full, &self.full[off..off + self.full_dim]);
                (id, s)
            })
            .collect();
        Ok(top_k(rescored, k))
    }
}

impl Retriever for MatryoshkaAdaptive {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, MarError> {
        self.search_full_query(query, k)
    }
    fn name(&self) -> &'static str { "matryoshka-adaptive" }
    fn resident_bytes(&self) -> usize { (self.full.len() + self.prefixes.len()) * 4 }
}

/// Parallel batch search helper. Returns one result list per query.
#[cfg(not(target_arch = "wasm32"))]
pub fn batch_search<R: Retriever + Sync>(
    r: &R,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Result<Vec<(u32, f32)>, MarError>> {
    queries.par_iter().map(|q| r.search(q, k)).collect()
}

#[cfg(target_arch = "wasm32")]
pub fn batch_search<R: Retriever>(
    r: &R,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Result<Vec<(u32, f32)>, MarError>> {
    queries.iter().map(|q| r.search(q, k)).collect()
}

/// Recall@k of `candidate` against `truth`. Both lists hold ids; scores ignored.
pub fn recall_at_k(truth: &[(u32, f32)], candidate: &[(u32, f32)], k: usize) -> f32 {
    let kk = k.min(truth.len()).min(candidate.len());
    if kk == 0 {
        return 0.0;
    }
    let truth_set: std::collections::HashSet<u32> =
        truth.iter().take(kk).map(|x| x.0).collect();
    let hits = candidate
        .iter()
        .take(kk)
        .filter(|x| truth_set.contains(&x.0))
        .count();
    hits as f32 / kk as f32
}

impl fmt::Debug for MatryoshkaAdaptive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatryoshkaAdaptive {{ n={}, full_dim={}, low_dim={}, rerank_factor={} }}",
            self.n, self.full_dim, self.low_dim, self.rerank_factor
        )
    }
}
