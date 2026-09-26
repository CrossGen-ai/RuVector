//! Batched-kernel search.
//!
//! Same per-query greedy traversal as [`crate::independent`], but every
//! candidate vector load is *fanned out* to the batched distance kernel:
//! we compute the distance from that candidate to **all Q queries** in one
//! pass. Only the query whose expansion produced this candidate uses the
//! result for frontier progress; the other queries' distance values are
//! discarded here (see [`crate::shared`] for the variant that reuses them).
//!
//! Why bother if we throw distances away? Because the memory cost of the
//! candidate vector load is amortized in the kernel — the same cache line
//! that services query `qi` also services `qj`. On correlated batches this
//! keeps the frontier expansions cheap even when they do not overlap.

use crate::dist::l2_sq_batch;
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

pub struct BatchedKernelSearch;

impl BatchedKernelSearch {
    /// Returns (top-k per query, total distinct candidate vector loads).
    pub fn search(
        graph: &ProximityGraph,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> (Vec<TopK>, u64) {
        let q = queries.len();
        let ef = ef.max(k);
        let mut visited: Vec<Vec<bool>> = (0..q).map(|_| vec![false; graph.n]).collect();
        let mut candidates: Vec<BinaryHeap<std::cmp::Reverse<HeapItem>>> =
            (0..q).map(|_| BinaryHeap::new()).collect();
        let mut results: Vec<BinaryHeap<HeapItem>> =
            (0..q).map(|_| BinaryHeap::new()).collect();
        let mut buf = vec![0.0f32; q];
        let mut loads: u64 = 0;

        // Seed all queries with the entry point via a single batched call.
        let entry = graph.entry;
        l2_sq_batch(graph.vector(entry), queries, &mut buf);
        loads += 1;
        for qi in 0..q {
            visited[qi][entry as usize] = true;
            let d = buf[qi];
            candidates[qi].push(std::cmp::Reverse(HeapItem { dist: d, id: entry }));
            results[qi].push(HeapItem { dist: d, id: entry });
        }

        // Round-robin over queries so cache reuse patterns line up.
        let mut active = vec![true; q];
        while active.iter().any(|&a| a) {
            for qi in 0..q {
                if !active[qi] { continue; }
                let Some(std::cmp::Reverse(c)) = candidates[qi].pop() else {
                    active[qi] = false; continue;
                };
                let worst = results[qi].peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                if c.dist > worst && results[qi].len() >= ef {
                    active[qi] = false; continue;
                }
                for &nid in graph.neighbours(c.id) {
                    if nid == u32::MAX { continue; }
                    let ni = nid as usize;
                    if visited[qi][ni] { continue; }
                    visited[qi][ni] = true;
                    // Batched distance kernel — computes for every query at
                    // once even though only `qi` progresses its frontier.
                    l2_sq_batch(graph.vector(nid), queries, &mut buf);
                    loads += 1;
                    let d = buf[qi];
                    let worst = results[qi].peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                    if results[qi].len() < ef || d < worst {
                        candidates[qi].push(std::cmp::Reverse(HeapItem { dist: d, id: nid }));
                        results[qi].push(HeapItem { dist: d, id: nid });
                        if results[qi].len() > ef { results[qi].pop(); }
                    }
                }
            }
        }

        let outs: Vec<TopK> = results.into_iter().map(|h| {
            let mut all = h.into_sorted_vec();
            all.truncate(k);
            TopK {
                ids: all.iter().map(|x| x.id).collect(),
                dists: all.iter().map(|x| x.dist).collect(),
            }
        }).collect();
        (outs, loads)
    }
}

impl BatchedSearch for BatchedKernelSearch {
    fn name(&self) -> &'static str { "batched-kernel" }
    fn search_batch(
        &self,
        graph: &ProximityGraph,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<TopK> {
        Self::search(graph, queries, k, ef).0
    }
}
