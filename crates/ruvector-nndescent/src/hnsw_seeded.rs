//! Seeded HNSW base layer: take an externally-built kNN graph and use
//! it directly as the layer-0 adjacency of an HNSW index.
//!
//! For this PoC we evaluate the kNN graph *as a single-layer NSW* —
//! i.e. greedy beam search with `ef` candidates starting from a
//! deterministic entry point. This isolates the build-quality
//! contribution of the kNN-graph builder (the variable under test)
//! from the upper-layer routing logic, which is identical across
//! all variants.
//!
//! The minimal HNSW already lives in `crates/ruvector-matryoshka`;
//! we deliberately don't duplicate it here. The seeded base-layer
//! demonstrates that a graph built bulk-style yields the same
//! search-quality contract as one built incrementally.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::distance::l2_sq;
use crate::knn_graph::KnnGraph;

#[derive(Clone, Debug)]
pub struct SeededHnswConfig {
    pub ef_search: usize,
    /// Multi-entry-point beam search: standard practice for graph indexes
    /// that lack upper-layer hierarchical routing. floor(log2(N))+1 random
    /// entries is the EFANNA/NSG default — restores navigability when
    /// the kNN graph is layer-0-only.
    pub entry_points: Vec<u32>,
}

impl Default for SeededHnswConfig {
    fn default() -> Self {
        Self { ef_search: 64, entry_points: vec![0] }
    }
}

impl SeededHnswConfig {
    /// Deterministic spread of `n_entries` entry points across `n` vectors.
    pub fn spread_entries(n: usize, n_entries: usize) -> Vec<u32> {
        let n_entries = n_entries.max(1).min(n);
        (0..n_entries)
            .map(|i| ((i as u64 * n as u64) / n_entries as u64) as u32)
            .collect()
    }
}

pub struct SeededHnsw<'a> {
    pub vectors: &'a [Vec<f32>],
    pub graph: &'a KnnGraph,
    pub cfg: SeededHnswConfig,
}

impl<'a> SeededHnsw<'a> {
    pub fn new(vectors: &'a [Vec<f32>], graph: &'a KnnGraph, cfg: SeededHnswConfig) -> Self {
        Self { vectors, graph, cfg }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<u32> {
        let ef = self.cfg.ef_search.max(k);
        let mut visited: HashSet<u32> = HashSet::new();
        let mut open: BinaryHeap<MinC> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxC> = BinaryHeap::new();

        let entries: &[u32] = if self.cfg.entry_points.is_empty() {
            &[0u32]
        } else {
            &self.cfg.entry_points
        };
        for &ep in entries {
            if !visited.insert(ep) { continue; }
            let ep_d = l2_sq(query, &self.vectors[ep as usize]);
            open.push(MinC { dist: ep_d, id: ep });
            results.push(MaxC { dist: ep_d, id: ep });
            if results.len() > ef { results.pop(); }
        }

        while let Some(curr) = open.pop() {
            let worst = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
            if curr.dist > worst { break; }
            for n in &self.graph.neighbors[curr.id as usize] {
                let nb = n.id;
                if !visited.insert(nb) { continue; }
                let d = l2_sq(query, &self.vectors[nb as usize]);
                let worst = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
                if d < worst || results.len() < ef {
                    open.push(MinC { dist: d, id: nb });
                    results.push(MaxC { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = results.into_iter().map(|c| (c.dist, c.id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

#[derive(Clone)]
struct MinC { dist: f32, id: u32 }
impl PartialEq for MinC { fn eq(&self, o: &Self) -> bool { self.id == o.id } }
impl Eq for MinC {}
impl PartialOrd for MinC { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MinC {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal).then(self.id.cmp(&o.id))
    }
}

#[derive(Clone)]
struct MaxC { dist: f32, id: u32 }
impl PartialEq for MaxC { fn eq(&self, o: &Self) -> bool { self.id == o.id } }
impl Eq for MaxC {}
impl PartialOrd for MaxC { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MaxC {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal).then(o.id.cmp(&self.id))
    }
}
