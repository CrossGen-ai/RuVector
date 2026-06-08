//! Greedy best-first traversal used by both queries and insertions.
//!
//! Maintains a fixed-capacity `eps`-best frontier and a visited bitmap. The
//! search converges when the frontier head is closer than every candidate's
//! neighbour, which mirrors the original DEG search proof.

use crate::distance::Metric;
use crate::graph::{Deg, NodeId};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One search result: (id, distance to query).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: NodeId,
    pub dist: f32,
}

impl Eq for SearchResult {}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // For a min-heap of "best so far": smaller dist == better.
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Wrapper that flips ordering for `BinaryHeap` (max-heap → min-heap of `MaxFirst`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct MaxFirst(SearchResult);
impl Eq for MaxFirst {}
impl Ord for MaxFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so largest-dist comes out first (used to evict frontier).
        self.0.dist.partial_cmp(&other.0.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MaxFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MinFirst(SearchResult);
impl Eq for MinFirst {}
impl Ord for MinFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Greedy search returning the top-`k` results. `eps` is the search width
/// (a.k.a. `ef`); larger == higher recall, more distance computations.
pub fn search<M: Metric>(
    deg: &Deg,
    query: &[f32],
    k: usize,
    eps: usize,
    entry: NodeId,
) -> Vec<SearchResult> {
    let n = deg.len();
    if n == 0 {
        return Vec::new();
    }
    let mut visited = vec![false; n];
    visited[entry as usize] = true;

    let entry_v = deg.vector(entry);
    let d0 = M::dist(query, entry_v);
    let seed = SearchResult { id: entry, dist: d0 };

    // Best-known results (max-heap on dist so we can evict the worst).
    let mut best: BinaryHeap<MaxFirst> = BinaryHeap::with_capacity(eps + 1);
    best.push(MaxFirst(seed));
    // Candidates to expand (min-heap on dist).
    let mut candidates: BinaryHeap<MinFirst> = BinaryHeap::with_capacity(eps + 1);
    candidates.push(MinFirst(seed));

    while let Some(MinFirst(cur)) = candidates.pop() {
        // If the closest unexpanded candidate is further than our worst
        // accepted neighbour, we cannot improve — bail.
        if let Some(&MaxFirst(worst)) = best.peek() {
            if cur.dist > worst.dist && best.len() >= eps {
                break;
            }
        }
        for &nb in deg.neighbours(cur.id) {
            let nb_us = nb as usize;
            if visited[nb_us] {
                continue;
            }
            visited[nb_us] = true;
            let d = M::dist(query, deg.vector(nb));
            let r = SearchResult { id: nb, dist: d };
            let take = if best.len() < eps {
                true
            } else {
                // peek must exist since len == eps > 0
                best.peek().map(|m| d < m.0.dist).unwrap_or(true)
            };
            if take {
                best.push(MaxFirst(r));
                candidates.push(MinFirst(r));
                while best.len() > eps {
                    best.pop();
                }
            }
        }
    }

    let mut out: Vec<SearchResult> = best.into_iter().map(|m| m.0).collect();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    out.truncate(k);
    out
}
