//! SeRF — Segment-Graph Range-Filtered Approximate Nearest Neighbor Search.
//!
//! Given a dataset of vectors each tagged with a scalar key (timestamp, price,
//! rating, …), answer queries of the form: "k nearest neighbors of q among
//! items with key in [lo, hi]".
//!
//! Three strategies live in this crate behind a small trait so they can be
//! benchmarked head-to-head with identical recall accounting:
//!
//! 1. [`flat::Flat`]              — brute-force baseline (postfilter).
//! 2. [`nsw_post::NswPost`]       — single global NSW + postfilter.
//! 3. [`segment::SegmentGraph`]   — segment-tree of NSW graphs (the SeRF approach).
//!
//! Distance is squared L2; all vectors are `Vec<f32>`. Real numbers, no mocks.

pub mod flat;
pub mod nsw;
pub mod nsw_post;
pub mod segment;

#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
}

impl Range {
    pub fn contains(&self, x: f32) -> bool {
        x >= self.lo && x <= self.hi
    }
}

/// Common interface for range-filtered ANN backends.
pub trait RangeAnn {
    fn search(&self, q: &[f32], range: Range, k: usize) -> Vec<(usize, f32)>;
    fn name(&self) -> &'static str;
}

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Compute Recall@k of `candidate` against `truth`.
pub fn recall(candidate: &[(usize, f32)], truth: &[(usize, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut hits = 0usize;
    for (id, _) in candidate.iter().take(truth.len()) {
        if truth.iter().any(|(t, _)| t == id) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}
