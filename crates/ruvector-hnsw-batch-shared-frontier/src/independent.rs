//! Baseline: one independent greedy `ef`-search per query.
//!
//! Standard HNSW inner-layer search:
//!
//! 1. Seed the candidate min-heap with the graph's entry point.
//! 2. Pop the closest unexpanded candidate. If it's farther than the
//!    current `k`-th best result, stop.
//! 3. Otherwise expand it, pushing unvisited neighbours whose distance
//!    beats the current result-heap's worst.
//!
//! No sharing between queries: each query re-loads every candidate vector
//! from memory, which is exactly the cost the batched backends amortize.

use crate::dist::l2_sq;
use crate::graph::ProximityGraph;
use crate::{BatchedSearch, TopK};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct HeapItem { dist: f32, id: u32 }

impl PartialEq for HeapItem { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for HeapItem {}
impl PartialOrd for HeapItem { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for HeapItem { fn cmp(&self, o: &Self) -> Ordering { self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal) } }

pub struct IndependentSearch;

impl IndependentSearch {
    /// Track total per-node vector loads for accounting (used by benches).
    pub fn search_one(
        graph: &ProximityGraph,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> (TopK, u64) {
        let ef = ef.max(k);
        let mut visited = vec![false; graph.n];
        // Min-heap over (-dist) — use Reverse.
        let mut candidates: BinaryHeap<std::cmp::Reverse<HeapItem>> = BinaryHeap::new();
        // Max-heap of current best-k results by distance.
        let mut results: BinaryHeap<HeapItem> = BinaryHeap::new();
        let mut loads: u64 = 0;

        let entry = graph.entry;
        visited[entry as usize] = true;
        let d0 = l2_sq(query, graph.vector(entry));
        loads += 1;
        candidates.push(std::cmp::Reverse(HeapItem { dist: d0, id: entry }));
        results.push(HeapItem { dist: d0, id: entry });

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = results.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef { break; }
            for &nid in graph.neighbours(c.id) {
                if nid == u32::MAX { continue; }
                let ni = nid as usize;
                if visited[ni] { continue; }
                visited[ni] = true;
                let d = l2_sq(query, graph.vector(nid));
                loads += 1;
                let worst = results.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(std::cmp::Reverse(HeapItem { dist: d, id: nid }));
                    results.push(HeapItem { dist: d, id: nid });
                    if results.len() > ef { results.pop(); }
                }
            }
        }

        let mut all: Vec<HeapItem> = results.into_sorted_vec();
        all.truncate(k);
        (TopK {
            ids: all.iter().map(|h| h.id).collect(),
            dists: all.iter().map(|h| h.dist).collect(),
        }, loads)
    }
}

impl BatchedSearch for IndependentSearch {
    fn name(&self) -> &'static str { "independent" }
    fn search_batch(
        &self,
        graph: &ProximityGraph,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<TopK> {
        queries.iter().map(|q| Self::search_one(graph, q, k, ef).0).collect()
    }
}
