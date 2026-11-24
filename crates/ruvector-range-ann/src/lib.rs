//! Range (epsilon-radius) ANN variants.
//!
//! Most ANN work targets top-k. Range ANN is a different retrieval mode:
//! given a query vector `q` and radius `r`, return every point `p` such that
//! `dist(q, p) <= r`. The result-set size is unknown a priori, which breaks
//! heap-bounded search assumptions and demands different early-termination
//! rules.
//!
//! This crate exposes a trait-based swappable design with three backends:
//!
//! - [`linear::LinearScan`] — baseline brute force scan.
//! - [`ivf::IvfRange`] — IVF index that only probes centroids whose
//!   envelope intersects the query ball.
//! - [`hnsw_expand::HnswExpand`] — HNSW graph where expansion is driven by
//!   an in-ball frontier rather than a top-k heap; a point enters the
//!   result set when its distance to `q` is `<= r`, and the frontier is
//!   extended as long as it contains any point within `r * (1 + alpha)`.
//!
//! Every backend implements [`RangeIndex`], making them drop-in swappable
//! for benchmarks or downstream code.

pub mod data;
pub mod hnsw_expand;
pub mod ivf;
pub mod linear;

use std::time::Duration;

/// A single range-search result: `(point_id, squared_euclidean_distance)`.
pub type RangeHit = (u32, f32);

/// Trait every range-ANN backend implements.
pub trait RangeIndex: Send + Sync {
    /// Build the index from the given points.
    fn build(points: &[Vec<f32>]) -> Self
    where
        Self: Sized;

    /// Return every point id whose squared L2 distance to `query` is `<= r2`.
    ///
    /// `r2` is the squared radius so the inner loop can skip a sqrt.
    /// The returned vector is in no guaranteed order.
    fn range_search(&self, query: &[f32], r2: f32) -> Vec<RangeHit>;

    /// Human-readable backend name used in benchmark reports.
    fn name(&self) -> &'static str;
}

/// Compute squared L2 distance between two equal-length slices.
#[inline(always)]
pub fn sqr_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Per-query benchmark measurement.
#[derive(Debug, Clone)]
pub struct QueryMeasurement {
    pub elapsed: Duration,
    pub hits: usize,
    /// Recall vs the exact ground truth (linear scan). 1.0 for the baseline.
    pub recall: f32,
}

/// Aggregate benchmark result for one backend.
#[derive(Debug, Clone)]
pub struct BackendReport {
    pub name: &'static str,
    pub build_time: Duration,
    pub per_query: Vec<QueryMeasurement>,
}

impl BackendReport {
    pub fn mean_query_us(&self) -> f64 {
        let n = self.per_query.len().max(1) as f64;
        let total_us: f64 = self.per_query.iter().map(|m| m.elapsed.as_secs_f64() * 1e6).sum();
        total_us / n
    }

    pub fn mean_hits(&self) -> f64 {
        let n = self.per_query.len().max(1) as f64;
        let total: f64 = self.per_query.iter().map(|m| m.hits as f64).sum();
        total / n
    }

    pub fn mean_recall(&self) -> f32 {
        let n = self.per_query.len().max(1) as f32;
        self.per_query.iter().map(|m| m.recall).sum::<f32>() / n
    }
}
