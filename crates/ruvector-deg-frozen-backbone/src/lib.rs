//! Dynamic Exploration Graph (DEG) with a frozen Delaunay-approximation backbone.
//!
//! Inspired by Neuhaus & Schubert, "Dynamic Exploration Graph" (2024), which
//! separates stable, quality-controlled edges from opportunistic exploration
//! edges. This crate is a reference PoC for the ruvector 2026-09-19 nightly.
//!
//! Three trait-swappable index variants share [`AnnIndex`]:
//!
//! * [`HnswLite`]         — greedy baseline, single-layer, RNG-pruned.
//! * [`MutableDeg`]       — DEG occlusion swap on every edge.
//! * [`FrozenBackboneDeg`] — leading fraction `α` of every neighbor list is
//!   frozen at insert time; the mutable tail follows DEG swap-if-better.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub type NodeId = u32;

mod deg;
mod hnsw_lite;
mod util;

pub use deg::{DegCore, FrozenBackboneDeg, MutableDeg};
pub use hnsw_lite::HnswLite;
pub use util::{brute_force_topk, make_clustered_dataset, recall_at_k, SplitMix64};

/// Squared L2 distance. Kept simple so it inlines cleanly for small `d`.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

pub trait AnnIndex {
    fn insert(&mut self, id: NodeId, v: Vec<f32>);
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)>;
    fn n_edges(&self) -> usize;
    fn n_nodes(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Flat vector store used by every variant.
pub struct VecStore {
    pub dim: usize,
    pub data: Vec<f32>,
    pub n: usize,
}
impl VecStore {
    pub fn new(dim: usize) -> Self { Self { dim, data: Vec::new(), n: 0 } }
    pub fn push(&mut self, v: &[f32]) -> NodeId {
        assert_eq!(v.len(), self.dim);
        let id = self.n as NodeId;
        self.data.extend_from_slice(v);
        self.n += 1;
        id
    }
    pub fn get(&self, id: NodeId) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }
}

#[derive(Clone, Copy)]
struct Cand { d: f32, id: NodeId }
impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.d == o.d && self.id == o.id } }
impl Eq for Cand {}
impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering { self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal) }
}

/// Best-first search returning up to `ef` best (id, d²) pairs sorted ascending.
pub(crate) fn beam_search<F>(
    entry: NodeId,
    query: &[f32],
    ef: usize,
    store: &VecStore,
    neighbors_of: F,
) -> Vec<(NodeId, f32)>
where
    F: Fn(NodeId, &mut dyn FnMut(NodeId)),
{
    let n_nodes = store.n;
    let mut visited = vec![false; n_nodes];
    let d0 = l2_sq(query, store.get(entry));
    #[derive(Clone, Copy)]
    struct Rev(Cand);
    impl PartialEq for Rev { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
    impl Eq for Rev {}
    impl PartialOrd for Rev { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
    impl Ord for Rev { fn cmp(&self, o: &Self) -> Ordering { o.0.cmp(&self.0) } }
    let mut frontier: BinaryHeap<Rev> = BinaryHeap::new();
    let mut best: BinaryHeap<Cand> = BinaryHeap::new();
    visited[entry as usize] = true;
    frontier.push(Rev(Cand { d: d0, id: entry }));
    best.push(Cand { d: d0, id: entry });

    while let Some(Rev(cur)) = frontier.pop() {
        if best.len() >= ef {
            let worst = best.peek().unwrap().d;
            if cur.d > worst { break; }
        }
        let mut cb = |nb: NodeId| {
            let ns = nb as usize;
            if visited[ns] { return; }
            visited[ns] = true;
            let d = l2_sq(query, store.get(nb));
            let worst = best.peek().map(|c| c.d).unwrap_or(f32::INFINITY);
            if best.len() < ef || d < worst {
                frontier.push(Rev(Cand { d, id: nb }));
                best.push(Cand { d, id: nb });
                if best.len() > ef { best.pop(); }
            }
        };
        neighbors_of(cur.id, &mut cb);
    }
    let mut out: Vec<(NodeId, f32)> = best.into_iter().map(|c| (c.id, c.d)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_all(n: usize, dim: usize) -> (VecStore, HnswLite, MutableDeg, FrozenBackboneDeg) {
        let data = make_clustered_dataset(n, dim, 8, 42);
        let mut store = VecStore::new(dim);
        for v in &data { store.push(v); }
        let mut h = HnswLite::new(dim, 16, 64);
        let mut mdeg = MutableDeg::new(dim, 16, 64);
        let mut fbdeg = FrozenBackboneDeg::new(dim, 16, 64, 0.25);
        for (i, v) in data.iter().enumerate() {
            h.insert(i as u32, v.clone());
            mdeg.insert(i as u32, v.clone());
            fbdeg.insert(i as u32, v.clone());
        }
        (store, h, mdeg, fbdeg)
    }

    #[test]
    fn recall_reaches_target() {
        let (store, h, m, f) = build_all(800, 32);
        let mut ok = [0u32; 3];
        for qi in (0..20).map(|i| store.n - 1 - i) {
            let q = store.get(qi as u32).to_vec();
            let truth = brute_force_topk(&store, &q, 10);
            for (idx, r) in [
                h.search(&q, 10, 64),
                m.search(&q, 10, 64),
                f.search(&q, 10, 64),
            ].iter().enumerate() {
                let rc = recall_at_k(r, &truth, 10);
                if rc >= 0.75 { ok[idx] += 1; }
            }
        }
        assert!(ok[0] >= 15, "hnsw-lite recall regressed: {}/20", ok[0]);
        assert!(ok[1] >= 15, "deg-mutable recall regressed: {}/20", ok[1]);
        assert!(ok[2] >= 15, "deg-frozen recall regressed: {}/20", ok[2]);
    }

    #[test]
    fn frozen_prefix_is_immutable() {
        let dim = 16;
        let mut f = FrozenBackboneDeg::new(dim, 8, 32, 0.5);
        let data = make_clustered_dataset(400, dim, 4, 7);
        for (i, v) in data.iter().enumerate().take(200) {
            f.insert(i as u32, v.clone());
        }
        let snap: Vec<Vec<NodeId>> = (0..f.0.neighbors.len())
            .map(|i| f.0.neighbors[i][..f.0.frozen_prefix[i]].to_vec())
            .collect();
        for (i, v) in data.iter().enumerate().skip(200) {
            f.insert(i as u32, v.clone());
        }
        for (i, s) in snap.iter().enumerate() {
            assert_eq!(&f.0.neighbors[i][..s.len()], &s[..], "frozen prefix of node {i} changed");
        }
    }
}
