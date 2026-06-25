//! ruvector-serf — Segment-graph Range-Filtered ANN (SeRF-style).
//!
//! Implements three backends with a shared [`RangeAnn`] trait, suitable for
//! head-to-head benchmarking on range-filtered nearest-neighbor queries:
//!
//! 1. [`linear::LinearPrefilter`]       — exact baseline, brute-force.
//! 2. [`post_filter::PostFilterNsw`]    — greedy graph + post-filter on attribute.
//! 3. [`serf::SerfIndex`]               — SeRF-lite: graph with attribute-aware
//!    edge pruning (skip edges to out-of-range nodes during traversal).
//!
//! Distance is squared L2. Vectors are `f32`. Each point carries a single
//! ordinal attribute (e.g. timestamp index), and queries take a closed range
//! `[lo, hi]` on that attribute.
//!
//! Inspired by SeRF (Zuo et al., SIGMOD 2024) and iRangeGraph (VLDB 2024).
//! This crate keeps the design simple enough to study trade-offs honestly:
//! files stay under 500 lines, every benchmark number in the research doc
//! comes from `cargo run --release -p ruvector-serf --bin serf-bench`.

pub mod data;
pub mod graph;
pub mod linear;
pub mod post_filter;
pub mod serf;

pub use data::{Dataset, Point, Query};

/// Search hit returned by all backends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

/// Common trait for range-filtered ANN backends. Returns up to `k` hits whose
/// attribute lies in the inclusive range `[query.lo, query.hi]`, ordered by
/// ascending squared-L2 distance.
pub trait RangeAnn {
    fn name(&self) -> &'static str;
    fn search(&self, query: &Query, k: usize) -> Vec<Hit>;
}

/// Compute recall@k of `candidate` against `truth` (both top-k lists for the
/// same query). Returns a value in `[0.0, 1.0]`.
pub fn recall_at_k(truth: &[Hit], candidate: &[Hit], k: usize) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let k = k.min(truth.len()).min(candidate.len().max(1));
    let truth_ids: std::collections::HashSet<u32> =
        truth.iter().take(k).map(|h| h.id).collect();
    let hits = candidate
        .iter()
        .take(k)
        .filter(|h| truth_ids.contains(&h.id))
        .count();
    hits as f32 / k as f32
}
