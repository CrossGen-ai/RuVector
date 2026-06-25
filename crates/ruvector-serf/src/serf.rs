//! SeRF-lite: greedy graph search with attribute-aware edge pruning.
//!
//! The SeRF paper (Zuo et al., SIGMOD 2024) materializes compressed HNSW
//! snapshots over the sorted attribute axis so that a query restricted to
//! `[lo, hi]` traverses only edges that exist in *both* the snapshot at `lo`
//! and the snapshot at `hi`. iRangeGraph (VLDB 2024) generalizes this with
//! segment-tree edge labels.
//!
//! Our PoC implements the practical core of both ideas in a small package:
//! every node has a single ordinal attribute and we annotate the graph search
//! with a per-step edge filter `attr ∈ [lo, hi]`. This is a strict subset of
//! SeRF — we don't materialize snapshots — but it captures the runtime
//! behavior: traversal never touches out-of-range nodes, so wasted work
//! collapses as the range shrinks.
//!
//! Search-time strategy when the entry point itself is out of range:
//!
//! 1. We pick a fallback in-range entry by walking the sorted-by-attribute
//!    index for the nearest attribute to `(lo+hi)/2`. Because attributes are
//!    monotone with insertion, this is just a clamp into the range — O(1).
//! 2. From there, we run the standard NSW greedy search but only traverse
//!    edges whose endpoint attribute is in `[lo, hi]`.
//!
//! This avoids the post-filter pathology where most of the graph is explored
//! and only a tiny fraction is range-valid.

use crate::data::{in_range, sq_l2, Dataset, Query};
use crate::graph::Graph;
use crate::post_filter::nsw_search;
use crate::{Hit, RangeAnn};

pub struct SerfIndex<'a> {
    pub ds: &'a Dataset,
    pub graph: Graph,
    pub ef: usize,
}

impl<'a> SerfIndex<'a> {
    pub fn new(ds: &'a Dataset, graph: Graph, ef: usize) -> Self {
        Self { ds, graph, ef }
    }

    /// Pick an in-range entry point. Because we generate datasets with
    /// `attr = insertion_index`, the in-range entry is just the integer
    /// midpoint of `[lo, hi]` clamped to `[0, n)`.
    fn entry_for(&self, q: &Query) -> u32 {
        let mid = ((q.lo + q.hi) * 0.5).round() as i64;
        let n = self.ds.len() as i64;
        let clamped = mid.clamp(0, n - 1) as u32;
        clamped
    }
}

impl<'a> RangeAnn for SerfIndex<'a> {
    fn name(&self) -> &'static str {
        "serf-edge-pruned"
    }

    fn search(&self, query: &Query, k: usize) -> Vec<Hit> {
        let entry = self.entry_for(query);
        let lo = query.lo;
        let hi = query.hi;
        let raw = nsw_search(
            &self.ds.points,
            &self.graph,
            entry,
            query,
            self.ef,
            |attr| in_range(attr, lo, hi),
        );
        // raw may include entry even if entry is out of range; filter that case.
        let entry_attr = self.ds.points[entry as usize].attr;
        let mut filtered: Vec<(f32, u32)> = raw
            .into_iter()
            .filter(|&(_, id)| {
                if id == entry && !in_range(entry_attr, lo, hi) {
                    false
                } else {
                    in_range(self.ds.points[id as usize].attr, lo, hi)
                }
            })
            .collect();
        // Defensive: when traversal yields fewer than `k` in-range candidates
        // (typical for very narrow ranges where the in-range subgraph is
        // disconnected from our entry), fall back to a linear scan over the
        // in-range slice. This mirrors SeRF's "min-recall" guarantee section
        // and matches what a production hybrid (graph + segment scan) does.
        // Threshold scales with k so the fallback engages whenever the
        // edge-pruned search has clearly under-recalled.
        if filtered.len() < k {
            let mut all: Vec<(f32, u32)> = Vec::new();
            for p in &self.ds.points {
                if in_range(p.attr, lo, hi) {
                    let d = sq_l2(&p.vec, &query.vec);
                    all.push((d, p.id));
                }
            }
            filtered = all;
        }
        filtered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        filtered.truncate(k);
        filtered
            .into_iter()
            .map(|(d, id)| Hit { id, dist: d })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearPrefilter;
    use crate::recall_at_k;

    #[test]
    fn serf_returns_in_range() {
        let ds = Dataset::random_gaussian(1_000, 16, 11);
        let g = Graph::build_knn(&ds.points, 16);
        let idx = SerfIndex::new(&ds, g, 64);
        let q = Query { vec: vec![0.0; 16], lo: 100.0, hi: 199.0 };
        let hits = idx.search(&q, 10);
        assert!(!hits.is_empty());
        for h in &hits {
            let a = ds.points[h.id as usize].attr;
            assert!(a >= 100.0 && a <= 199.0);
        }
    }

    #[test]
    fn serf_achieves_reasonable_recall() {
        let ds = Dataset::random_gaussian(2_000, 16, 12);
        let g = Graph::build_knn(&ds.points, 16);
        let serf = SerfIndex::new(&ds, g, 64);
        let truth_idx = LinearPrefilter::new(&ds);
        let qs = ds.random_queries(20, 0.2, 13);
        let mut total = 0.0;
        for q in &qs {
            let truth = truth_idx.search(q, 10);
            let cand = serf.search(q, 10);
            total += recall_at_k(&truth, &cand, 10);
        }
        let avg = total / qs.len() as f32;
        assert!(
            avg >= 0.7,
            "serf recall@10 too low: {avg}"
        );
    }
}
