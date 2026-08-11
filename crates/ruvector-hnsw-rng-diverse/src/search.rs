//! Best-first beam search with configurable `ef`. Returns top-k plus stats.

use crate::{dist2, graph::GraphIndex};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct Item {
    dist: f32,
    id: usize,
}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap() }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.dist.partial_cmp(&other.dist) }
}
impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist }
}
impl Eq for Item {}

#[derive(Default, Debug, Clone)]
pub struct SearchStats {
    pub dist_calls: usize,
    pub hops: usize,
}

/// Convenience wrapper: single-entry search.
pub fn search(
    idx: &GraphIndex,
    vectors: &[Vec<f32>],
    entry: usize,
    query: &[f32],
    k: usize,
    ef: usize,
) -> (Vec<(usize, f32)>, SearchStats) {
    search_multi(idx, vectors, &[entry], query, k, ef)
}

/// Multi-entry beam search. Seeds the candidate/result heaps from every entry
/// so a single-layer graph that is disconnected across dense clusters can
/// still cover the space with a handful of diverse restarts.
pub fn search_multi(
    idx: &GraphIndex,
    vectors: &[Vec<f32>],
    entries: &[usize],
    query: &[f32],
    k: usize,
    ef: usize,
) -> (Vec<(usize, f32)>, SearchStats) {
    let n = vectors.len();
    let mut visited = vec![false; n];
    let mut stats = SearchStats::default();

    let mut candidates: BinaryHeap<std::cmp::Reverse<Item>> = BinaryHeap::new();
    let mut results: BinaryHeap<Item> = BinaryHeap::new();

    for &e in entries {
        if visited[e] { continue; }
        let d = dist2(&vectors[e], query);
        stats.dist_calls += 1;
        visited[e] = true;
        candidates.push(std::cmp::Reverse(Item { dist: d, id: e }));
        results.push(Item { dist: d, id: e });
        if results.len() > ef { results.pop(); }
    }

    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        // If the closest candidate is worse than the worst kept result, stop.
        if let Some(worst) = results.peek() {
            if current.dist > worst.dist && results.len() >= ef {
                break;
            }
        }
        stats.hops += 1;
        for &nid in &idx.adj[current.id] {
            if visited[nid] { continue; }
            visited[nid] = true;
            let d = dist2(&vectors[nid], query);
            stats.dist_calls += 1;
            let worst = results.peek().map(|w| w.dist).unwrap_or(f32::INFINITY);
            if results.len() < ef || d < worst {
                candidates.push(std::cmp::Reverse(Item { dist: d, id: nid }));
                results.push(Item { dist: d, id: nid });
                if results.len() > ef { results.pop(); }
            }
        }
    }

    let mut out: Vec<(usize, f32)> = results.into_iter().map(|i| (i.id, i.dist)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.truncate(k);
    (out, stats)
}
