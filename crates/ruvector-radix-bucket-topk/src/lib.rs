//! Heap-less top-K candidate ordering for ANN beam search.
//!
//! ADR-362. See `docs/research/nightly/2026-11-19-radix-bucket-topk/README.md`
//! for a walkthrough of the design and the measured trade-off.
//!
//! The `TopKCollector` trait abstracts over the three ordering strategies a
//! beam-search worker can use:
//!
//! * [`BinaryHeapTopK`] - std binary max-heap, O(log k) push.
//! * [`SortedVecTopK`]  - unbounded then sort, O(1) push / O(n log n) finalize.
//! * [`RadixBucketTopK`] - exponentially-spaced radix buckets, O(1) push, O(k)
//!   finalize (with an in-bucket sort).
//!
//! The radix variant is the subject of this nightly: it swaps worst-case
//! ordering precision for cache-friendly O(1) inserts, which matters when the
//! beam explores many candidates that get pruned before finalization.

#![forbid(unsafe_code)]

use std::cmp::Ordering;

pub mod extract;
pub mod radix;

pub use radix::RadixBucketTopK;

/// A (candidate id, distance) pair. Distance is always non-negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: u32,
    pub dist: f32,
}

impl Candidate {
    #[inline]
    pub fn new(id: u32, dist: f32) -> Self {
        debug_assert!(dist.is_finite() && dist >= 0.0, "distance must be non-negative and finite");
        Self { id, dist }
    }
}

/// Ordering by distance ascending, breaking ties by id for determinism.
fn cmp_cand(a: &Candidate, b: &Candidate) -> Ordering {
    a.dist
        .partial_cmp(&b.dist)
        .unwrap_or(Ordering::Equal)
        .then(a.id.cmp(&b.id))
}

/// Trait implemented by every top-K ordering strategy benchmarked in this
/// crate. Collectors are consumed by `finalize` so the same object cannot be
/// reused after its contents are materialized.
pub trait TopKCollector {
    /// Insert a candidate. Implementations MUST remain safe when `dist` is
    /// `+inf`; they MAY assume it is finite for efficiency.
    fn push(&mut self, cand: Candidate);

    /// Current upper bound on the K-th smallest distance. Beam search uses this
    /// to prune without visiting a candidate's underlying vector. Returns
    /// `f32::INFINITY` while the collector has strictly fewer than K entries.
    fn kth_upper_bound(&self) -> f32;

    /// Number of candidates currently retained. Collectors that trim
    /// themselves eagerly may return a value less than the total `push` count.
    fn len(&self) -> usize;

    /// Materialize the final top-K in ascending distance order.
    fn finalize(self) -> Vec<Candidate>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Binary max-heap backed collector (classic ANN implementation).
pub struct BinaryHeapTopK {
    k: usize,
    heap: std::collections::BinaryHeap<HeapEntry>,
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry(Candidate);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        cmp_cand(&self.0, &other.0) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap keyed by distance so we can pop the worst candidate
        // whenever we exceed K.
        cmp_cand(&self.0, &other.0)
    }
}

impl BinaryHeapTopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k + 1),
        }
    }
}

impl TopKCollector for BinaryHeapTopK {
    fn push(&mut self, cand: Candidate) {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry(cand));
            return;
        }
        // SAFETY: len() >= k > 0 so peek is Some.
        let worst = self.heap.peek().unwrap().0;
        if cmp_cand(&cand, &worst) == Ordering::Less {
            self.heap.pop();
            self.heap.push(HeapEntry(cand));
        }
    }

    fn kth_upper_bound(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().unwrap().0.dist
        }
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn finalize(self) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = self.heap.into_iter().map(|h| h.0).collect();
        out.sort_by(cmp_cand);
        out
    }
}

/// Grow-unbounded collector that sorts at `finalize`. Simple baseline that
/// trades memory for push throughput.
pub struct SortedVecTopK {
    k: usize,
    buf: Vec<Candidate>,
}

impl SortedVecTopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            k,
            buf: Vec::with_capacity(k * 8),
        }
    }
}

impl TopKCollector for SortedVecTopK {
    #[inline]
    fn push(&mut self, cand: Candidate) {
        self.buf.push(cand);
    }

    fn kth_upper_bound(&self) -> f32 {
        // No eager pruning; a correct upper bound would require an extra pass
        // every push. The beam-search caller that wants pruning must use the
        // heap or radix variant.
        f32::INFINITY
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn finalize(mut self) -> Vec<Candidate> {
        self.buf.sort_by(cmp_cand);
        self.buf.truncate(self.k);
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Candidate> {
        [
            (0u32, 0.5f32),
            (1, 0.1),
            (2, 0.9),
            (3, 0.3),
            (4, 0.7),
            (5, 0.2),
            (6, 0.6),
            (7, 0.05),
            (8, 0.95),
            (9, 0.4),
        ]
        .iter()
        .map(|(id, d)| Candidate::new(*id, *d))
        .collect()
    }

    fn exact_top3() -> Vec<Candidate> {
        vec![
            Candidate::new(7, 0.05),
            Candidate::new(1, 0.10),
            Candidate::new(5, 0.20),
        ]
    }

    #[test]
    fn binary_heap_matches_exact() {
        let mut c = BinaryHeapTopK::new(3);
        for s in sample() {
            c.push(s);
        }
        assert_eq!(c.finalize(), exact_top3());
    }

    #[test]
    fn sorted_vec_matches_exact() {
        let mut c = SortedVecTopK::new(3);
        for s in sample() {
            c.push(s);
        }
        assert_eq!(c.finalize(), exact_top3());
    }

    #[test]
    fn heap_upper_bound_tightens() {
        let mut c = BinaryHeapTopK::new(3);
        assert_eq!(c.kth_upper_bound(), f32::INFINITY);
        for s in sample() {
            c.push(s);
        }
        // After seeing all samples the 3rd best is 0.20
        assert!((c.kth_upper_bound() - 0.20).abs() < 1e-6);
    }
}
