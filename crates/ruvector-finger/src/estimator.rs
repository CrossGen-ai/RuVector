//! Traits shared by all FINGER-style distance estimators.
//!
//! The design intentionally mirrors the way an HNSW/ACORN search visits a
//! *pivot* (the current best node) and asks "which of your neighbours are the
//! most promising?". Each estimator returns a lightweight per-pivot handle so
//! query-time projections are amortised across all neighbour evaluations.

use std::sync::atomic::{AtomicU64, Ordering};

/// Handle returned by [`DistanceEstimator::prepare_query`] for a single pivot.
///
/// A handle stores any per-pivot query state (e.g. `q · pivot`, the projected
/// residual `B^T q_res`) so scoring each of the pivot's `M` neighbours becomes
/// a small dot-product rather than a full O(d) inner product.
pub trait PivotHandle {
    fn score(&self, neighbour: u32) -> f32;
}

/// Estimator abstraction used by the search loop.
///
/// Implementors keep whatever precomputed structure they need (raw residuals,
/// per-pivot bases, JL projections, …) and expose a `prepare_query` entry
/// point that returns a boxed [`PivotHandle`].
pub trait DistanceEstimator: Sync {
    /// Human-readable label for reports (`"exact"`, `"finger-r16"`, …).
    fn name(&self) -> &'static str;
    /// Persistent bytes required to store the estimator (excluding raw vectors
    /// unless the estimator retains them). Reported alongside benchmarks.
    fn bytes_per_vector(&self) -> usize;
    /// Return a query-scoped handle for `pivot_id`.
    fn prepare_query<'a>(&'a self, query: &[f32], pivot_id: u32) -> Box<dyn PivotHandle + 'a>;
}

/// Simple counter used by benchmarks to attribute cost to the estimator.
#[derive(Default)]
pub struct SearchStats {
    scored: AtomicU64,
    exact_rerank: AtomicU64,
}

impl SearchStats {
    pub fn record_scored(&self, n: u64) { self.scored.fetch_add(n, Ordering::Relaxed); }
    pub fn record_rerank(&self, n: u64) { self.exact_rerank.fetch_add(n, Ordering::Relaxed); }
    pub fn scored(&self) -> u64 { self.scored.load(Ordering::Relaxed) }
    pub fn rerank(&self) -> u64 { self.exact_rerank.load(Ordering::Relaxed) }
}
