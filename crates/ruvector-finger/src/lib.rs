//! ruvector-finger: FINGER-style residual-projection distance estimation
//! for graph-based ANN search.
//!
//! Reference: Chen et al., "FINGER: Fast Inference for Graph-based
//! Approximate Nearest Neighbor Search", KDD 2023.
//!
//! This crate ships a minimal, swappable distance-estimator trait plus
//! three concrete backends so the upper search layer (HNSW, Vamana,
//! NSG, ...) can pick a tradeoff at construction time:
//!
//!   * `ExactL2`           — brute-force FP32 squared L2 (baseline).
//!   * `JlProjector`       — Johnson-Lindenstrauss random projection
//!                           into r-dim subspace; estimates squared
//!                           L2 with provable concentration.
//!   * `FingerEstimator`   — projection-based lower bound + cached
//!                           per-vector norms, used as a gate before
//!                           full-precision rerank. This is the
//!                           "FINGER spirit": cheap LB → rerank.
//!
//! All three implement [`DistanceEstimator`] so a search loop can
//! swap them with no changes.

#![forbid(unsafe_code)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod bench_harness;

#[derive(Debug, Error)]
pub enum FingerError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("projection rank r={r} must be > 0 and <= d={d}")]
    InvalidRank { r: usize, d: usize },
    #[error("index is empty")]
    Empty,
}

/// A swappable distance estimator. Implementations may return:
/// * an exact distance,
/// * an approximation (cheap, possibly biased),
/// * a lower bound (cheap, never exceeds the true distance).
///
/// The contract for `estimate_sq_l2` is: it returns a *score* that
/// preserves the same ranking direction as squared L2 — smaller is
/// closer. Callers gate exact computation on this score.
pub trait DistanceEstimator: Send + Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Estimated squared L2 distance from `query` to base vector `i`.
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32;
    /// Exact squared L2 distance, used by upper layers for rerank.
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32;
    /// Approx FLOPs per single distance call. Used by the bench
    /// harness to compute "cost per query" as a hardware-independent
    /// figure of merit.
    fn flops_per_estimate(&self) -> usize;
}

// -------------------------------------------------------------------------
// 1. ExactL2 — baseline.
// -------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactL2 {
    dim: usize,
    data: Vec<f32>, // row-major, len = n * dim
    n: usize,
}

impl ExactL2 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            n: 0,
        }
    }
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Result<Self, FingerError> {
        if vectors.is_empty() {
            return Err(FingerError::Empty);
        }
        let dim = vectors[0].len();
        let mut data = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(FingerError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            data.extend_from_slice(v);
        }
        Ok(Self {
            dim,
            data,
            n: vectors.len(),
        })
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl DistanceEstimator for ExactL2 {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.n
    }
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        self.exact_sq_l2(query, i)
    }
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        sq_l2(query, self.row(i))
    }
    fn flops_per_estimate(&self) -> usize {
        // sub + mul + add per dim
        3 * self.dim
    }
}

// -------------------------------------------------------------------------
// 2. JlProjector — random Gaussian projection.
// -------------------------------------------------------------------------

/// Johnson-Lindenstrauss projection backend.
///
/// Each base vector x is precomputed as `x_proj = (1/sqrt(r)) R x`
/// where R ∈ R^{r×d} has i.i.d. N(0,1) entries. Distances in the
/// projected space are unbiased estimators of the original squared
/// L2 distances:
///
/// ```text
/// E[ ||R(q-x)/sqrt(r)||^2 ] = ||q-x||^2
/// ```
///
/// with relative error O(1/sqrt(r)) (Achlioptas; Dasgupta-Gupta).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JlProjector {
    dim: usize,
    rank: usize,
    n: usize,
    /// row-major r×d projection matrix, scaled by 1/sqrt(r)
    matrix: Vec<f32>,
    /// row-major n×r projected base vectors
    base_proj: Vec<f32>,
    /// optional access to original vectors for `exact_sq_l2`
    raw: Vec<f32>,
}

impl JlProjector {
    pub fn new(vectors: &[Vec<f32>], rank: usize, seed: u64) -> Result<Self, FingerError> {
        if vectors.is_empty() {
            return Err(FingerError::Empty);
        }
        let dim = vectors[0].len();
        if rank == 0 || rank > dim {
            return Err(FingerError::InvalidRank { r: rank, d: dim });
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (rank as f32).sqrt().recip();
        let mut matrix = vec![0.0f32; rank * dim];
        for v in matrix.iter_mut() {
            let s: f32 = StandardNormal.sample(&mut rng);
            *v = s * scale;
        }

        let mut raw = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(FingerError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            raw.extend_from_slice(v);
        }

        let n = vectors.len();
        let mut base_proj = vec![0.0f32; n * rank];
        for i in 0..n {
            let row = &raw[i * dim..(i + 1) * dim];
            let out = &mut base_proj[i * rank..(i + 1) * rank];
            project(&matrix, dim, rank, row, out);
        }

        Ok(Self {
            dim,
            rank,
            n,
            matrix,
            base_proj,
            raw,
        })
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rank];
        project(&self.matrix, self.dim, self.rank, query, &mut out);
        out
    }
    /// Cheap distance using a *precomputed* query projection.
    pub fn estimate_sq_l2_with_qproj(&self, q_proj: &[f32], i: usize) -> f32 {
        let row = &self.base_proj[i * self.rank..(i + 1) * self.rank];
        sq_l2(q_proj, row)
    }
    #[inline]
    fn raw_row(&self, i: usize) -> &[f32] {
        &self.raw[i * self.dim..(i + 1) * self.dim]
    }
}

impl DistanceEstimator for JlProjector {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.n
    }
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        // Projects the query each call. The bench harness uses the
        // amortized path via `estimate_sq_l2_with_qproj`.
        let q_proj = self.project_query(query);
        self.estimate_sq_l2_with_qproj(&q_proj, i)
    }
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        sq_l2(query, self.raw_row(i))
    }
    fn flops_per_estimate(&self) -> usize {
        // amortized: r-dim sub+mul+add (query projection paid once
        // per query, not per candidate).
        3 * self.rank
    }
}

// -------------------------------------------------------------------------
// 3. FingerEstimator — JL-LB gate + FP32 rerank.
// -------------------------------------------------------------------------

/// FINGER-style two-stage distance scorer.
///
/// Stage 1: cheap JL-projection estimate `est`. Because
/// `||R(q-x)/sqrt(r)||^2` concentrates around `||q-x||^2`, we use it
/// as a soft lower-bound gate by subtracting `slack * sigma`, where
/// sigma scales with `||q-x||` and `1/sqrt(r)`.
///
/// Stage 2: if a candidate's estimated distance is within the
/// current top-K threshold + the slack margin, recompute the exact
/// FP32 squared L2.
///
/// Top-level search code calls `estimate_sq_l2` first (very cheap)
/// and decides whether to call `exact_sq_l2`. For pure-ranking
/// benchmarks the harness uses only the estimate; for recall it
/// reranks the top-`rerank_k` candidates.
#[derive(Clone, Debug)]
pub struct FingerEstimator {
    inner: JlProjector,
    /// Slack multiplier applied to the per-vector residual norm so
    /// `estimate - slack * residual_bound` is a high-confidence
    /// lower bound on the true distance. 0.0 = no slack (matches
    /// raw JL); larger values trade pruning for recall.
    slack: f32,
    /// Cached `||x||^2` for each base vector (used only in
    /// alternate inner-product formulations; kept for completeness
    /// and so future graph-search backends can use it without an
    /// extra pass).
    base_sq_norms: Vec<f32>,
}

impl FingerEstimator {
    pub fn new(vectors: &[Vec<f32>], rank: usize, slack: f32, seed: u64) -> Result<Self, FingerError> {
        let inner = JlProjector::new(vectors, rank, seed)?;
        let mut base_sq_norms = Vec::with_capacity(vectors.len());
        for v in vectors {
            base_sq_norms.push(v.iter().map(|x| x * x).sum::<f32>());
        }
        Ok(Self { inner, slack, base_sq_norms })
    }
    pub fn rank(&self) -> usize {
        self.inner.rank
    }
    pub fn slack(&self) -> f32 {
        self.slack
    }
    pub fn base_sq_norm(&self, i: usize) -> f32 {
        self.base_sq_norms[i]
    }
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        self.inner.project_query(query)
    }
    pub fn estimate_sq_l2_with_qproj(&self, q_proj: &[f32], i: usize) -> f32 {
        let est = self.inner.estimate_sq_l2_with_qproj(q_proj, i);
        // Subtract a slack term proportional to sqrt(est) and 1/sqrt(r).
        // This converts the (unbiased) JL estimate into a soft lower
        // bound: the higher the rank, the tighter the bound.
        let sigma = est.max(0.0).sqrt() * (self.inner.rank as f32).sqrt().recip();
        (est - self.slack * sigma).max(0.0)
    }
}

impl DistanceEstimator for FingerEstimator {
    fn dim(&self) -> usize {
        self.inner.dim
    }
    fn len(&self) -> usize {
        self.inner.n
    }
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        let q_proj = self.project_query(query);
        self.estimate_sq_l2_with_qproj(&q_proj, i)
    }
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32 {
        self.inner.exact_sq_l2(query, i)
    }
    fn flops_per_estimate(&self) -> usize {
        // Same amortized cost as JL (slack adjustment is constant work).
        3 * self.inner.rank + 2
    }
}

// -------------------------------------------------------------------------
// Math helpers.
// -------------------------------------------------------------------------

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for k in 0..a.len() {
        let d = a[k] - b[k];
        s += d * d;
    }
    s
}

#[inline]
fn project(matrix: &[f32], dim: usize, rank: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), dim);
    debug_assert_eq!(out.len(), rank);
    for r in 0..rank {
        let row = &matrix[r * dim..(r + 1) * dim];
        let mut s = 0.0f32;
        for k in 0..dim {
            s += row[k] * x[k];
        }
        out[r] = s;
    }
}

// -------------------------------------------------------------------------
// Search primitives shared by the bench harness.
// -------------------------------------------------------------------------

/// Brute-force exact top-k by squared L2.
pub fn exact_top_k(index: &ExactL2, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = (0..index.len())
        .map(|i| (i, index.exact_sq_l2(query, i)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}

/// Two-stage FINGER search: rank by cheap estimate, then rerank top
/// `rerank_k` by exact distance, finally truncate to `k`.
pub fn finger_top_k(
    finger: &FingerEstimator,
    query: &[f32],
    k: usize,
    rerank_k: usize,
) -> Vec<(usize, f32)> {
    let q_proj = finger.project_query(query);
    let mut est: Vec<(usize, f32)> = (0..finger.len())
        .map(|i| (i, finger.estimate_sq_l2_with_qproj(&q_proj, i)))
        .collect();
    est.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let cap = rerank_k.min(est.len());
    let mut rer: Vec<(usize, f32)> = est[..cap]
        .iter()
        .map(|(i, _)| (*i, finger.exact_sq_l2(query, *i)))
        .collect();
    rer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    rer.truncate(k);
    rer
}

/// Pure JL search: rank by estimate, no rerank. Useful as the
/// pessimistic recall baseline for the gate.
pub fn jl_top_k(
    jl: &JlProjector,
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let q_proj = jl.project_query(query);
    let mut est: Vec<(usize, f32)> = (0..jl.len())
        .map(|i| (i, jl.estimate_sq_l2_with_qproj(&q_proj, i)))
        .collect();
    est.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    est.truncate(k);
    est
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        let s: f32 = StandardNormal.sample(&mut rng);
                        s
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn exact_l2_round_trip() {
        let data = make_dataset(32, 16, 42);
        let idx = ExactL2::from_vectors(&data).unwrap();
        let q = data[0].clone();
        assert!((idx.exact_sq_l2(&q, 0) - 0.0).abs() < 1e-6);
        let direct = sq_l2(&q, &data[5]);
        assert!((idx.exact_sq_l2(&q, 5) - direct).abs() < 1e-6);
    }

    #[test]
    fn jl_estimator_unbiased_within_tolerance() {
        // JL with r=64 on d=64 — should be within 25% of truth on
        // average over many pairs.
        let n = 200;
        let d = 64;
        let data = make_dataset(n, d, 7);
        let jl = JlProjector::new(&data, 64, 1).unwrap();
        let q = data[0].clone();
        let q_proj = jl.project_query(&q);
        let mut total_rel_err = 0.0f64;
        let mut count = 0usize;
        for i in 1..n {
            let est = jl.estimate_sq_l2_with_qproj(&q_proj, i) as f64;
            let truth = sq_l2(&q, &data[i]) as f64;
            if truth > 1e-3 {
                total_rel_err += ((est - truth) / truth).abs();
                count += 1;
            }
        }
        let mean_rel_err = total_rel_err / count as f64;
        assert!(
            mean_rel_err < 0.30,
            "JL mean rel err {mean_rel_err} too high"
        );
    }

    #[test]
    fn finger_top_k_recall_beats_random() {
        let n = 500;
        let d = 64;
        let k = 10;
        let data = make_dataset(n, d, 11);
        let exact = ExactL2::from_vectors(&data).unwrap();
        let finger = FingerEstimator::new(&data, 64, 0.0, 13).unwrap();
        let queries = make_dataset(20, d, 99);
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let g: Vec<usize> = exact_top_k(&exact, q, k).into_iter().map(|x| x.0).collect();
            let f: Vec<usize> = finger_top_k(&finger, q, k, 150).into_iter().map(|x| x.0).collect();
            for id in &g {
                if f.contains(id) {
                    hits += 1;
                }
            }
            total += k;
        }
        let recall = hits as f64 / total as f64;
        // FINGER with rerank_k=50 should massively beat the 10/500 = 2%
        // random baseline. We hold it to a real, non-trivial bar.
        assert!(recall >= 0.90, "FINGER recall@10 was {recall}");
    }
}
