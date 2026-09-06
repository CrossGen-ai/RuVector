//! Reverse-KNN Verified Retrieval.
//!
//! Idea: after a base ANN pass returns `M` candidates for query `q`, keep only
//! those candidates `c` whose *own* k-nearest neighborhood in the dataset
//! contains `q` (or, in the offline variant, contains some prototype near `q`).
//!
//! Intuition: an asymmetric-neighborhood point that appears near `q` only because
//! it is a hub / high-degree outlier is filtered out, since `q` will typically
//! not appear in `c`'s own tight neighborhood.
//!
//! This crate provides:
//!   * [`FlatL2Index`] — reference brute-force index over `f32` vectors.
//!   * [`NnIndex`] trait — pluggable search backend for the base pass.
//!   * [`RknnVerifier`] — post-hoc verifier with two modes:
//!       * `Live`  : compute each candidate's own top-K on the fly.
//!       * `Cached`: use a pre-built reverse-KNN adjacency list.
//!
//! No mocks. All numbers in the benchmark are `cargo run --release` outputs.

use std::cmp::Ordering;

pub mod dataset;
pub mod verify;

pub use verify::{RknnVerifier, VerifyMode};

/// A search backend that returns `(id, squared_distance)` pairs.
pub trait NnIndex: Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn vector(&self, id: usize) -> &[f32];
    /// Return the top-`k` neighbors by squared L2 distance to `q`, sorted asc.
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)>;
}

/// Squared Euclidean distance.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Reference flat (brute-force) L2 index. Deterministic, exact, cache-friendly.
pub struct FlatL2Index {
    dim: usize,
    data: Vec<f32>, // row-major, len = n * dim
    n: usize,
}

impl FlatL2Index {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
            n: 0,
        }
    }

    pub fn from_rows(dim: usize, rows: &[Vec<f32>]) -> Self {
        let mut data = Vec::with_capacity(rows.len() * dim);
        for r in rows {
            assert_eq!(r.len(), dim);
            data.extend_from_slice(r);
        }
        Self {
            dim,
            data,
            n: rows.len(),
        }
    }

    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
        self.n += 1;
    }
}

impl NnIndex for FlatL2Index {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.n
    }

    fn vector(&self, id: usize) -> &[f32] {
        let start = id * self.dim;
        &self.data[start..start + self.dim]
    }

    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        debug_assert_eq!(q.len(), self.dim);
        // Bounded max-heap of size k keyed by distance.
        let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
        let cmp = |a: &(f32, usize), b: &(f32, usize)| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        };
        for id in 0..self.n {
            let d = sq_l2(q, self.vector(id));
            if heap.len() < k {
                heap.push((d, id));
                if heap.len() == k {
                    heap.sort_by(cmp);
                }
            } else if d < heap[k - 1].0 {
                heap[k - 1] = (d, id);
                // Re-place the worst by bubble-up (small k, cheap).
                let mut i = k - 1;
                while i > 0 && heap[i].0 < heap[i - 1].0 {
                    heap.swap(i, i - 1);
                    i -= 1;
                }
            }
        }
        heap.sort_by(cmp);
        heap.into_iter().map(|(d, id)| (id, d)).collect()
    }
}

/// A noisy-ANN wrapper that simulates realistic recall loss for benchmark
/// variants without needing a heavy external HNSW dependency. It preserves
/// determinism by seeded shuffling of a super-set of exact candidates and
/// re-ordering them, then keeping the top-`M`.
///
/// This is used to produce a *baseline* whose recall is < 1.0, so that the
/// verifier's precision-lift over the baseline is measurable.
pub struct NoisyAnn<'a> {
    inner: &'a FlatL2Index,
    superset_mult: usize,
    seed: u64,
}

impl<'a> NoisyAnn<'a> {
    pub fn new(inner: &'a FlatL2Index, superset_mult: usize, seed: u64) -> Self {
        Self {
            inner,
            superset_mult,
            seed,
        }
    }
}

impl<'a> NnIndex for NoisyAnn<'a> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn vector(&self, id: usize) -> &[f32] {
        self.inner.vector(id)
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        let super_k = (k * self.superset_mult).min(self.inner.len());
        let mut cands = self.inner.search(q, super_k);
        // Deterministic pseudo-random perturbation: add a hash-derived jitter
        // to distances, then re-rank. This mimics a coarse quantizer that
        // sometimes misranks near-ties.
        for (id, d) in cands.iter_mut() {
            let mut h = self.seed
                .wrapping_add((*id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            let jitter = ((h & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.35 * (*d + 1e-3);
            *d += jitter;
        }
        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        cands.truncate(k);
        cands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dataset() -> FlatL2Index {
        let rows: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                let x = i as f32;
                vec![x.sin(), x.cos(), (x * 0.5).sin(), (x * 0.5).cos()]
            })
            .collect();
        FlatL2Index::from_rows(4, &rows)
    }

    #[test]
    fn flat_search_returns_query_first() {
        let idx = tiny_dataset();
        let q = idx.vector(7).to_vec();
        let hits = idx.search(&q, 5);
        assert_eq!(hits[0].0, 7);
        assert!(hits[0].1 < 1e-6);
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn sq_l2_is_zero_for_equal() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(sq_l2(&a, &a) < 1e-9);
    }

    #[test]
    fn noisy_ann_reorders_but_stays_within_superset() {
        let idx = tiny_dataset();
        let noisy = NoisyAnn::new(&idx, 4, 42);
        let q = idx.vector(3).to_vec();
        let exact: Vec<usize> = idx.search(&q, 20).into_iter().map(|(i, _)| i).collect();
        let noisy_top: Vec<usize> = noisy.search(&q, 5).into_iter().map(|(i, _)| i).collect();
        for id in &noisy_top {
            assert!(exact.contains(id), "noisy id {id} not in exact top-20");
        }
    }
}
