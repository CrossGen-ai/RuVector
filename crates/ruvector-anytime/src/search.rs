//! Three beam-search variants demonstrating the anytime emission policy.
//!
//! All variants share the same priority-queue beam-search skeleton. They
//! differ only in **when** the current top-k snapshot is emitted to the
//! caller via the `on_snapshot` callback.
//!
//! ## Monotonicity guarantee
//!
//! The skeleton maintains the top-k results in a max-heap that only ever
//! replaces the farthest element with a strictly closer one. Therefore the
//! multiset of distances in the result set is *non-increasing in
//! Pareto-distance terms* over time. Concretely:
//!
//! * The **farthest** distance in any snapshot is `≤` the farthest in all
//!   prior snapshots once the heap reaches size k.
//! * The **best-of-k** (smallest) distance is non-increasing across snapshots.
//!
//! This is verified by [`tests::monotone_property_holds`].
//!
//! ## Variants
//!
//! | Variant            | Emission                                             |
//! |--------------------|------------------------------------------------------|
//! | `OneShotSearch`    | Final result only                                    |
//! | `NaiveAnytime`     | After every pop (high overhead)                      |
//! | `BatchedAnytime`   | On improvement, with exponentially growing batch     |

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::graph::{l2_sq, FlatGraph};

/// Ordered f32 wrapper for `BinaryHeap`.
#[derive(Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A snapshot of the current top-k delivered to the caller mid-search.
#[derive(Debug, Clone)]
pub struct AnytimeSnapshot {
    /// Current top-k as `(id, squared_L2)`, sorted nearest first.
    pub neighbors: Vec<(u32, f32)>,
    /// Wall-clock nanoseconds since search start when the snapshot was emitted.
    pub elapsed_ns: u128,
    /// Number of candidates popped from the heap so far.
    pub pops: usize,
}

/// Output from a single query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Final top-k as `(id, squared_L2)`, sorted nearest first.
    pub neighbors: Vec<(u32, f32)>,
    /// Total candidate pops.
    pub pops: usize,
    /// Total neighbor-expansions.
    pub expansions: usize,
    /// Number of intermediate snapshots emitted (excluding the final result).
    pub snapshots_emitted: usize,
}

/// Pluggable search backend.
///
/// `on_snapshot` is invoked zero or more times during the search; for
/// `OneShotSearch` it is never invoked. The final answer is always returned
/// in [`SearchResult::neighbors`].
pub trait Searcher {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
        on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
    ) -> SearchResult;
}

// ──────────────────────────────────────────────────────────────────────────────
// Emission policies
// ──────────────────────────────────────────────────────────────────────────────

/// Decides whether a snapshot should be emitted at the current step.
trait EmitPolicy {
    /// Called after the top-k heap is updated for the just-popped candidate.
    /// `best_improved` is true if the smallest distance in the top-k strictly
    /// decreased compared to before this step.
    fn should_emit(&mut self, pops: usize, best_improved: bool) -> bool;
}

/// Never emit intermediate snapshots — caller only gets the final result.
struct NeverEmit;
impl EmitPolicy for NeverEmit {
    #[inline]
    fn should_emit(&mut self, _pops: usize, _best_improved: bool) -> bool {
        false
    }
}

/// Emit on every pop (after the top-k update). High overhead, maximum
/// reactivity. Useful as an upper-bound on snapshot count for benchmarking.
struct AlwaysEmit;
impl EmitPolicy for AlwaysEmit {
    #[inline]
    fn should_emit(&mut self, _pops: usize, _best_improved: bool) -> bool {
        true
    }
}

/// Emit only when the best-of-k strictly improves AND the batch counter has
/// expired. Batch size grows exponentially after each emission, so early
/// search (when improvement is rapid) reports frequently and late search
/// (diminishing returns) reports rarely.
struct BatchedImprove {
    counter: usize,
    next_threshold: usize,
    growth: f32,
}

impl BatchedImprove {
    fn new(initial: usize, growth: f32) -> Self {
        BatchedImprove {
            counter: 0,
            next_threshold: initial,
            growth,
        }
    }
}

impl EmitPolicy for BatchedImprove {
    fn should_emit(&mut self, _pops: usize, best_improved: bool) -> bool {
        self.counter += 1;
        if best_improved && self.counter >= self.next_threshold {
            self.counter = 0;
            // Grow next batch — but always at least 1 step.
            self.next_threshold =
                ((self.next_threshold as f32 * self.growth).ceil() as usize).max(1);
            true
        } else {
            false
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Variants
// ──────────────────────────────────────────────────────────────────────────────

/// One-shot baseline — no intermediate snapshots.
pub struct OneShotSearch;
impl Searcher for OneShotSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
        on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
    ) -> SearchResult {
        beam_search(graph, query, k, ef, entry_id, NeverEmit, on_snapshot)
    }
}

/// Naive anytime — emits a snapshot after every pop. Maximally reactive,
/// maximally overhead-heavy.
pub struct NaiveAnytime;
impl Searcher for NaiveAnytime {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
        on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
    ) -> SearchResult {
        beam_search(graph, query, k, ef, entry_id, AlwaysEmit, on_snapshot)
    }
}

/// Batched anytime — emits on improvement with an exponentially growing batch.
///
/// `initial_batch` is the number of pops between the first two improvement
/// emissions; each subsequent emission requires `growth ×` more pops. With
/// `growth = 2.0` (default), early reactivity is preserved while late-search
/// overhead drops to near-zero.
pub struct BatchedAnytime {
    pub initial_batch: usize,
    pub growth: f32,
}

impl Default for BatchedAnytime {
    fn default() -> Self {
        BatchedAnytime {
            initial_batch: 1,
            growth: 2.0,
        }
    }
}

impl Searcher for BatchedAnytime {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
        on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
    ) -> SearchResult {
        let policy = BatchedImprove::new(self.initial_batch, self.growth.max(1.0));
        beam_search(graph, query, k, ef, entry_id, policy, on_snapshot)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core skeleton
// ──────────────────────────────────────────────────────────────────────────────

fn beam_search<P: EmitPolicy>(
    graph: &FlatGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    entry_id: usize,
    mut policy: P,
    on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
) -> SearchResult {
    let started = Instant::now();
    if graph.is_empty() {
        return SearchResult {
            neighbors: vec![],
            pops: 0,
            expansions: 0,
            snapshots_emitted: 0,
        };
    }

    let n = graph.n;
    let ef = ef.max(k);
    let entry = entry_id.min(n - 1);
    let entry_vec = graph.row(entry);

    let mut visited: Vec<bool> = vec![false; n];
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::with_capacity(ef + 1);
    // Max-heap of top-k results (peek = farthest accepted).
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(k + 1);

    let d0 = l2_sq(query, entry_vec);
    candidates.push(Reverse((OrdF32(d0), entry as u32)));
    visited[entry] = true;

    let mut pops = 0usize;
    let mut expansions = 0usize;
    let mut snapshots_emitted = 0usize;
    let mut best_so_far = f32::INFINITY;

    while let Some(Reverse((OrdF32(curr_d), curr))) = candidates.pop() {
        pops += 1;

        // Early stop: bounded beam termination.
        if results.len() >= k {
            if let Some(&(OrdF32(worst), _)) = results.peek() {
                if curr_d > worst {
                    break;
                }
            }
        }

        results.push((OrdF32(curr_d), curr));
        if results.len() > k {
            results.pop();
        }

        let new_best = curr_d.min(best_so_far);
        let best_improved = new_best < best_so_far;
        best_so_far = new_best;

        if policy.should_emit(pops, best_improved) {
            let snap = snapshot_from(&results, started.elapsed().as_nanos(), pops);
            on_snapshot(&snap);
            snapshots_emitted += 1;
        }

        // Expand neighbors.
        expansions += 1;
        let worst_pending = candidates
            .iter()
            .map(|Reverse((OrdF32(d), _))| *d)
            .fold(f32::NEG_INFINITY, f32::max);

        for &nbr in &graph.neighbors[curr as usize] {
            let ni = nbr as usize;
            if visited[ni] {
                continue;
            }
            visited[ni] = true;
            let nd = l2_sq(query, graph.row(ni));
            if candidates.len() < ef || nd < worst_pending {
                candidates.push(Reverse((OrdF32(nd), nbr)));
                if candidates.len() > ef {
                    let mut items: Vec<_> = candidates.drain().collect();
                    items.sort_unstable_by(|a, b| a.0 .0 .0.total_cmp(&b.0 .0 .0));
                    items.truncate(ef);
                    candidates.extend(items);
                }
            }
        }
    }

    let mut neighbors: Vec<(u32, f32)> = results
        .into_iter()
        .map(|(OrdF32(d), id)| (id, d))
        .collect();
    neighbors.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    SearchResult {
        neighbors,
        pops,
        expansions,
        snapshots_emitted,
    }
}

fn snapshot_from(
    results: &BinaryHeap<(OrdF32, u32)>,
    elapsed_ns: u128,
    pops: usize,
) -> AnytimeSnapshot {
    let mut v: Vec<(u32, f32)> = results
        .iter()
        .map(|&(OrdF32(d), id)| (id, d))
        .collect();
    v.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    AnytimeSnapshot {
        neighbors: v,
        elapsed_ns,
        pops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{clustered_queries, clustered_unit_vectors};
    use crate::graph::GraphConfig;

    fn build_test_graph() -> (FlatGraph, Vec<Vec<f32>>) {
        let (data, _) = clustered_unit_vectors(4, 60, 16, 0.12, 0xD00D);
        let g = FlatGraph::build(
            data.clone(),
            GraphConfig { m: 8, m_longjump: 4, dims: 16 },
        );
        let q = clustered_queries(8, 16, &data, 60, 0.12, 0xBEEF);
        (g, q)
    }

    /// **Monotone Top-k property.**
    /// For each pair of consecutive snapshots `(S_i, S_{i+1})`:
    ///   - if both reached size k, farthest(S_{i+1}) ≤ farthest(S_i)
    ///   - best-of-k of S_{i+1} ≤ best-of-k of S_i
    #[test]
    fn monotone_property_holds() {
        let (g, queries) = build_test_graph();
        for q in &queries {
            let mut snaps: Vec<AnytimeSnapshot> = vec![];
            let _ = NaiveAnytime.search(&g, q, 5, 40, 0, &mut |s| snaps.push(s.clone()));
            assert!(snaps.len() >= 2, "expected multiple snapshots");
            for w in snaps.windows(2) {
                let (a, b) = (&w[0], &w[1]);
                let best_a = a.neighbors.first().map(|x| x.1).unwrap_or(f32::INFINITY);
                let best_b = b.neighbors.first().map(|x| x.1).unwrap_or(f32::INFINITY);
                assert!(best_b <= best_a, "best-of-k regressed: {best_a} → {best_b}");
                if a.neighbors.len() == 5 && b.neighbors.len() == 5 {
                    let far_a = a.neighbors.last().unwrap().1;
                    let far_b = b.neighbors.last().unwrap().1;
                    assert!(
                        far_b <= far_a,
                        "farthest regressed: {far_a} → {far_b}"
                    );
                }
            }
        }
    }

    /// One-shot result must equal the result observed by a naive anytime
    /// run's final snapshot — the emission policy must not perturb the answer.
    #[test]
    fn anytime_does_not_change_final_result() {
        let (g, queries) = build_test_graph();
        for q in &queries {
            let r_one = OneShotSearch.search(&g, q, 5, 40, 0, &mut |_| {});
            let mut last_snap: Option<AnytimeSnapshot> = None;
            let r_any = NaiveAnytime.search(&g, q, 5, 40, 0, &mut |s| {
                last_snap = Some(s.clone());
            });
            assert_eq!(r_one.neighbors, r_any.neighbors);
            let last = last_snap.expect("naive emits at least once");
            assert_eq!(last.neighbors, r_one.neighbors);
        }
    }

    /// Batched policy must emit strictly fewer snapshots than naive,
    /// while preserving the final top-k.
    #[test]
    fn batched_emits_fewer_snapshots_than_naive() {
        let (g, queries) = build_test_graph();
        let batched = BatchedAnytime::default();
        let mut total_naive = 0usize;
        let mut total_batched = 0usize;
        for q in &queries {
            let r_n = NaiveAnytime.search(&g, q, 5, 40, 0, &mut |_| {});
            let r_b = batched.search(&g, q, 5, 40, 0, &mut |_| {});
            assert_eq!(r_n.neighbors, r_b.neighbors);
            total_naive += r_n.snapshots_emitted;
            total_batched += r_b.snapshots_emitted;
        }
        assert!(
            total_batched < total_naive,
            "batched ({total_batched}) >= naive ({total_naive})"
        );
    }
}
