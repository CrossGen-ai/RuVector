//! Beam search over a k-NN graph. Identical loop for every policy.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::KnnGraph;
use crate::policy::TerminationPolicy;
use crate::util::sq_l2;
use crate::Scored;

/// Per-query counters emitted by [`search`].
#[derive(Copy, Clone, Debug, Default)]
pub struct SearchStats {
    pub visits: usize,        // graph nodes popped from candidate heap
    pub distances: usize,     // distance computations against neighbors
    pub early_stopped: bool,
    pub final_ef: usize,      // size of the result heap when we stopped
}

/// Best-first beam search on `graph`.
///
/// The dynamics deliberately mirror HNSW's layer-0 search:
///
///   1. Push `entry_ids` onto the candidate min-heap and results max-heap.
///   2. Pop best candidate `c`. If `c.dist > worst-in-topk` and result heap
///      is already full to `ef_max`, stop (classic HNSW termination).
///   3. Otherwise expand `c`'s neighbors: for each unvisited neighbor
///      compute distance, push to both heaps (respecting `ef_max`).
///   4. After each expansion, snapshot the top-`k` and call the policy.
pub fn search(
    graph: &KnnGraph,
    query: &[f32],
    entry_ids: &[u32],
    k: usize,
    ef_max: usize,
    policy: &mut dyn TerminationPolicy,
) -> (Vec<Scored>, SearchStats) {
    assert_eq!(query.len(), graph.dim);
    assert!(ef_max >= k, "ef_max ({ef_max}) must be >= k ({k})");
    policy.reset();

    // Visited bitset — one bit per node.
    let mut visited = vec![false; graph.n];

    // candidates: min-heap ordered by distance ascending (via `Reverse`).
    let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(ef_max * 2);
    // results: max-heap ordered by distance descending, capped at `ef_max`.
    let mut results: BinaryHeap<Scored> = BinaryHeap::with_capacity(ef_max + 1);

    for &id in entry_ids {
        if visited[id as usize] { continue; }
        visited[id as usize] = true;
        let d = sq_l2(query, graph.vec(id));
        let s = Scored { dist: d, id };
        candidates.push(Reverse(s));
        results.push(s);
        if results.len() > ef_max { results.pop(); }
    }

    let mut stats = SearchStats { distances: entry_ids.len(), ..Default::default() };

    while let Some(Reverse(c)) = candidates.pop() {
        stats.visits += 1;
        // Classic HNSW termination: if the best remaining candidate is worse
        // than our current worst top-ef, further expansion cannot improve
        // top-k. (This is the same rule ADR-303 / adaptive-recall etc build
        // on.)
        if let Some(worst) = results.peek() {
            if c.dist > worst.dist && results.len() >= ef_max {
                break;
            }
        }

        for &nb in graph.neighbors(c.id) {
            let idx = nb as usize;
            if visited[idx] { continue; }
            visited[idx] = true;
            let d = sq_l2(query, graph.vec(nb));
            stats.distances += 1;
            let s = Scored { dist: d, id: nb };
            let admit = match results.peek() {
                Some(worst) => results.len() < ef_max || s.dist < worst.dist,
                None => true,
            };
            if admit {
                candidates.push(Reverse(s));
                results.push(s);
                if results.len() > ef_max { results.pop(); }
            }
        }

        // Snapshot current top-k (sorted ascending by distance) and let the
        // policy decide.
        let top_k = top_k_snapshot(&results, k);
        if policy.should_stop(stats.visits, k, &top_k) {
            stats.early_stopped = true;
            break;
        }
    }

    stats.final_ef = results.len();
    let sorted = top_k_snapshot(&results, k);
    (sorted, stats)
}

/// Snapshot the top-k of a max-heap of size ≤ ef_max, returned sorted
/// ascending by distance. O(ef log ef); ef is small in practice.
fn top_k_snapshot(results: &BinaryHeap<Scored>, k: usize) -> Vec<Scored> {
    let mut v: Vec<Scored> = results.iter().cloned().collect();
    v.sort();
    v.truncate(k);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::FixedBudget;
    use crate::util::random_unit_vectors;

    #[test]
    fn search_returns_k_ids() {
        let data = random_unit_vectors(200, 16, 11);
        let g = KnnGraph::build(data, 16, 12);
        let mut pol = FixedBudget;
        let (top, stats) = search(&g, g.vec(0), &[7, 42, 100], 10, 32, &mut pol);
        assert_eq!(top.len(), 10);
        assert!(stats.visits > 0);
    }

    #[test]
    fn search_recall_is_reasonable_on_synthetic() {
        // With ef=64 on a k-NN graph over 500 random unit vectors, recall@10
        // should be well above 0.5; we only sanity-check.
        let data = random_unit_vectors(500, 32, 13);
        let g = KnnGraph::build(data, 32, 16);
        let mut pol = FixedBudget;
        let q = g.vec(3).to_vec();
        let gt = g.brute_topk(&q, 10);
        let (top, _) = search(&g, &q, &[0, 100, 250, 400], 10, 64, &mut pol);
        let hit = top.iter().filter(|s| gt.contains(&s.id)).count();
        assert!(hit >= 5, "recall too low: {hit}/10");
    }
}
