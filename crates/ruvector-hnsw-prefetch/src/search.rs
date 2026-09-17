//! HNSW-style beam search with pluggable prefetch policy.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::graph::{squared_l2, FlatGraph};
use crate::prefetch::PrefetchPolicy;

#[derive(Copy, Clone, Debug)]
pub struct SearchStats {
    /// Full-precision distance calls performed.
    pub distance_calls: u64,
    /// Neighbours visited (may deduplicate against a bitset).
    pub visited: u64,
    /// Nodes accepted into the candidate heap (feedback signal used by
    /// the adaptive policy).
    pub accepted: u64,
    /// Final prefetch depth at end of query (only meaningful for
    /// adaptive backend).
    pub final_prefetch_depth: u32,
}

impl SearchStats {
    fn zero() -> Self {
        Self {
            distance_calls: 0,
            visited: 0,
            accepted: 0,
            final_prefetch_depth: 0,
        }
    }
}

/// Which traversal implementation to invoke.
#[derive(Copy, Clone, Debug)]
pub enum TraversalBackend {
    NoPrefetch,
    Prefetch1,
    Adaptive,
}

/// Beam search over `graph` returning the top-`k` nearest ids to `query`.
///
/// The candidate frontier is a min-heap on distance; the result set is a
/// max-heap of size at most `ef`. This is the textbook HNSW inner-layer
/// search.
pub fn search(
    graph: &FlatGraph,
    query: &[f32],
    k: usize,
    ef: usize,
    backend: TraversalBackend,
) -> (Vec<u32>, SearchStats) {
    let policy = match backend {
        TraversalBackend::NoPrefetch => PrefetchPolicy::None,
        TraversalBackend::Prefetch1 => PrefetchPolicy::Fixed(1),
        TraversalBackend::Adaptive => PrefetchPolicy::Adaptive { initial: 2, max: 6 },
    };

    let mut stats = SearchStats::zero();
    let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 8);
    let mut frontier: BinaryHeap<MinNode> = BinaryHeap::new();
    let mut results: BinaryHeap<MaxNode> = BinaryHeap::new();
    let mut depth = policy.initial_depth() as u32;
    let max_depth = policy.max_depth() as u32;

    // Seed with entry point.
    let entry = graph.entry;
    let d0 = squared_l2(query, graph.vector(entry));
    stats.distance_calls += 1;
    visited.insert(entry);
    frontier.push(MinNode { dist: d0, id: entry });
    results.push(MaxNode { dist: d0, id: entry });

    while let Some(cur) = frontier.pop() {
        // If the closest frontier node is farther than the worst kept
        // result and we already have `ef` candidates, we are done.
        if let Some(worst) = results.peek() {
            if results.len() >= ef && cur.dist > worst.dist {
                break;
            }
        }

        // Prefetch the CSR neighbour row itself; the immediate cost is
        // one cache line but it hides the dependent load.
        graph.prefetch_neighbor_row(cur.id);
        let neigh = graph.neighbors_of(cur.id);

        // Prefetch initial window of neighbour vectors.
        let d_used = depth.min(max_depth) as usize;
        for j in 0..d_used.min(neigh.len()) {
            graph.prefetch_vector_of(neigh[j]);
        }

        for (i, &nid) in neigh.iter().enumerate() {
            if !visited.insert(nid) {
                continue;
            }
            stats.visited += 1;

            // Prefetch the neighbour `i + d_used` ahead of full-distance
            // computation for neighbour `i`.
            let ahead = i + d_used;
            if ahead < neigh.len() {
                graph.prefetch_vector_of(neigh[ahead]);
            }

            let d = squared_l2(query, graph.vector(nid));
            stats.distance_calls += 1;

            let worst = results.peek().map(|w| w.dist).unwrap_or(f32::INFINITY);
            let accepted = results.len() < ef || d < worst;
            if accepted {
                stats.accepted += 1;
                frontier.push(MinNode { dist: d, id: nid });
                results.push(MaxNode { dist: d, id: nid });
                if results.len() > ef {
                    results.pop();
                }
                if policy.is_adaptive() && depth < max_depth {
                    depth += 1;
                }
            } else if policy.is_adaptive() && depth > 1 {
                depth -= 1;
            }
        }
    }

    stats.final_prefetch_depth = depth;

    // Extract top-k in ascending distance order.
    let mut out: Vec<(f32, u32)> = results.into_iter().map(|n| (n.dist, n.id)).collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out.truncate(k);
    (out.into_iter().map(|(_, id)| id).collect(), stats)
}

/// Brute-force ground truth used to compute recall.
pub fn brute_force(graph: &FlatGraph, query: &[f32], k: usize) -> Vec<u32> {
    let mut all: Vec<(f32, u32)> = (0..graph.n as u32)
        .map(|i| (squared_l2(query, graph.vector(i)), i))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all.truncate(k);
    all.into_iter().map(|(_, i)| i).collect()
}

// -- Heap adapters --

#[derive(Copy, Clone, Debug)]
struct MinNode {
    dist: f32,
    id: u32,
}
impl Eq for MinNode {}
impl PartialEq for MinNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Ord for MinNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip so BinaryHeap pops the smallest.
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, Debug)]
struct MaxNode {
    dist: f32,
    id: u32,
}
impl Eq for MaxNode {}
impl PartialEq for MaxNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Ord for MaxNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MaxNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// -- Unit tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphConfig;

    fn tiny_graph() -> FlatGraph {
        FlatGraph::build(GraphConfig {
            n: 1024,
            dim: 32,
            m: 8,
            seed: 7,
        })
    }

    #[test]
    fn all_backends_return_k_ids() {
        let g = tiny_graph();
        let q = g.vector(3).to_vec();
        for backend in [
            TraversalBackend::NoPrefetch,
            TraversalBackend::Prefetch1,
            TraversalBackend::Adaptive,
        ] {
            let (ids, stats) = search(&g, &q, 10, 32, backend);
            assert_eq!(ids.len(), 10, "backend {:?} returned wrong k", backend);
            assert!(stats.distance_calls > 0);
        }
    }

    #[test]
    fn adaptive_matches_baseline_result_set_size() {
        let g = tiny_graph();
        let q = g.vector(11).to_vec();
        let (a, _) = search(&g, &q, 10, 32, TraversalBackend::NoPrefetch);
        let (b, _) = search(&g, &q, 10, 32, TraversalBackend::Adaptive);
        assert_eq!(a.len(), b.len());
        // Prefetch policy MUST NOT change results (same order, same set).
        let sa: std::collections::HashSet<_> = a.iter().collect();
        let sb: std::collections::HashSet<_> = b.iter().collect();
        assert_eq!(sa, sb, "prefetch changed result set");
    }

    #[test]
    fn recall_is_nontrivial() {
        let g = tiny_graph();
        let q = g.vector(51).to_vec();
        let (ann, _) = search(&g, &q, 10, 64, TraversalBackend::Prefetch1);
        let bf = brute_force(&g, &q, 10);
        let ann_set: std::collections::HashSet<_> = ann.into_iter().collect();
        let bf_set: std::collections::HashSet<_> = bf.into_iter().collect();
        let overlap = ann_set.intersection(&bf_set).count();
        assert!(overlap >= 5, "recall too low: {}/10", overlap);
    }
}
