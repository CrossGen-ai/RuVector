//! Dynamic Exploration Graph (mutable) and Frozen-Backbone variant.
use crate::{beam_search, l2_sq, AnnIndex, NodeId, VecStore};
use std::cmp::Ordering;

pub struct DegCore {
    pub store: VecStore,
    pub m: usize,
    pub ef_build: usize,
    pub neighbors: Vec<Vec<NodeId>>,
    pub frozen_prefix: Vec<usize>,
    pub(crate) entry: Option<NodeId>,
    pub allow_rewire: bool,
    pub freeze_fraction: f32,
}

impl DegCore {
    pub(crate) fn new(dim: usize, m: usize, ef_build: usize, allow_rewire: bool, freeze_fraction: f32) -> Self {
        Self {
            store: VecStore::new(dim),
            m,
            ef_build,
            neighbors: Vec::new(),
            frozen_prefix: Vec::new(),
            entry: None,
            allow_rewire,
            freeze_fraction: freeze_fraction.clamp(0.0, 1.0),
        }
    }

    fn rng_prune(&self, cand: &mut Vec<(NodeId, f32)>, target: usize) {
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut kept: Vec<(NodeId, f32)> = Vec::with_capacity(target);
        'outer: for &(n, dq) in cand.iter() {
            for &(k, _) in kept.iter() {
                let dkn = l2_sq(self.store.get(n), self.store.get(k));
                if dkn < dq { continue 'outer; }
            }
            kept.push((n, dq));
            if kept.len() >= target { break; }
        }
        cand.clear();
        cand.extend(kept);
    }

    fn try_add_edge(&mut self, from: NodeId, to: NodeId) {
        let fs = from as usize;
        if self.neighbors[fs].iter().any(|&x| x == to) { return; }
        if self.neighbors[fs].len() < self.m {
            self.neighbors[fs].push(to);
            return;
        }
        if !self.allow_rewire { return; }
        let d_new = l2_sq(self.store.get(from), self.store.get(to));
        let start = self.frozen_prefix[fs];
        let mut worst_idx = None;
        let mut worst_d = -1.0f32;
        for i in start..self.neighbors[fs].len() {
            let nb = self.neighbors[fs][i];
            let d = l2_sq(self.store.get(from), self.store.get(nb));
            if d > worst_d { worst_d = d; worst_idx = Some(i); }
        }
        if let Some(i) = worst_idx {
            if d_new < worst_d { self.neighbors[fs][i] = to; }
        }
    }
}

impl AnnIndex for DegCore {
    fn name(&self) -> &'static str {
        if self.freeze_fraction > 0.0 { "deg-frozen-backbone" } else { "deg-mutable" }
    }
    fn n_nodes(&self) -> usize { self.store.n }
    fn n_edges(&self) -> usize { self.neighbors.iter().map(|v| v.len()).sum() }

    fn insert(&mut self, _id: NodeId, v: Vec<f32>) {
        let new_id = self.store.push(&v);
        self.neighbors.push(Vec::new());
        self.frozen_prefix.push(0);
        if self.entry.is_none() { self.entry = Some(new_id); return; }
        let entry = self.entry.unwrap();
        let neighbors_ref = &self.neighbors;
        let mut cand = beam_search(entry, &v, self.ef_build, &self.store, |cur, cb| {
            for &nb in &neighbors_ref[cur as usize] {
                if nb != new_id { cb(nb); }
            }
        });
        cand.retain(|(id, _)| *id != new_id);
        self.rng_prune(&mut cand, self.m);
        let n_freeze = ((cand.len() as f32) * self.freeze_fraction).round() as usize;
        for &(nb, _) in &cand {
            self.neighbors[new_id as usize].push(nb);
        }
        self.frozen_prefix[new_id as usize] = n_freeze.min(self.neighbors[new_id as usize].len());
        for &(nb, _) in &cand {
            self.try_add_edge(nb, new_id);
        }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let entry = match self.entry { Some(e) => e, None => return vec![] };
        let neighbors_ref = &self.neighbors;
        let mut out = beam_search(entry, q, ef, &self.store, |cur, cb| {
            for &nb in &neighbors_ref[cur as usize] { cb(nb); }
        });
        out.truncate(k);
        out
    }
}

pub struct MutableDeg(pub DegCore);
impl MutableDeg {
    pub fn new(dim: usize, m: usize, ef_build: usize) -> Self {
        Self(DegCore::new(dim, m, ef_build, true, 0.0))
    }
}
impl AnnIndex for MutableDeg {
    fn insert(&mut self, id: NodeId, v: Vec<f32>) { self.0.insert(id, v) }
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> { self.0.search(q, k, ef) }
    fn n_edges(&self) -> usize { self.0.n_edges() }
    fn n_nodes(&self) -> usize { self.0.n_nodes() }
    fn name(&self) -> &'static str { "deg-mutable" }
}

pub struct FrozenBackboneDeg(pub DegCore);
impl FrozenBackboneDeg {
    pub fn new(dim: usize, m: usize, ef_build: usize, freeze_fraction: f32) -> Self {
        Self(DegCore::new(dim, m, ef_build, true, freeze_fraction))
    }
}
impl AnnIndex for FrozenBackboneDeg {
    fn insert(&mut self, id: NodeId, v: Vec<f32>) { self.0.insert(id, v) }
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> { self.0.search(q, k, ef) }
    fn n_edges(&self) -> usize { self.0.n_edges() }
    fn n_nodes(&self) -> usize { self.0.n_nodes() }
    fn name(&self) -> &'static str { "deg-frozen-backbone" }
}

// Keep Ordering import used in future extensions; silence unused if not.
#[allow(dead_code)]
fn _keep_ordering() -> Ordering { Ordering::Equal }
