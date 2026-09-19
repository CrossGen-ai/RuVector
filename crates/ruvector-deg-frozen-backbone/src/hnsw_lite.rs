//! Single-layer HNSW-style baseline with RNG (Vamana α=1.0) pruning.
use crate::{beam_search, l2_sq, AnnIndex, NodeId, VecStore};
use std::cmp::Ordering;

pub struct HnswLite {
    pub store: VecStore,
    pub m: usize,
    pub ef_build: usize,
    pub neighbors: Vec<Vec<NodeId>>,
    pub(crate) entry: Option<NodeId>,
}

impl HnswLite {
    pub fn new(dim: usize, m: usize, ef_build: usize) -> Self {
        Self {
            store: VecStore::new(dim),
            m,
            ef_build,
            neighbors: Vec::new(),
            entry: None,
        }
    }

    fn rng_prune(&self, cand: &mut Vec<(NodeId, f32)>, target: usize) {
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut kept: Vec<(NodeId, f32)> = Vec::with_capacity(target);
        'outer: for &(n, dq) in cand.iter() {
            for &(k, _) in kept.iter() {
                let dkn = l2_sq(self.store.get(n), self.store.get(k));
                if dkn < dq {
                    continue 'outer;
                }
            }
            kept.push((n, dq));
            if kept.len() >= target {
                break;
            }
        }
        cand.clear();
        cand.extend(kept);
    }
}

impl AnnIndex for HnswLite {
    fn name(&self) -> &'static str { "hnsw-lite" }
    fn n_nodes(&self) -> usize { self.store.n }
    fn n_edges(&self) -> usize { self.neighbors.iter().map(|v| v.len()).sum() }

    fn insert(&mut self, _id: NodeId, v: Vec<f32>) {
        let new_id = self.store.push(&v);
        self.neighbors.push(Vec::new());
        if self.entry.is_none() {
            self.entry = Some(new_id);
            return;
        }
        let entry = self.entry.unwrap();
        let neighbors_ref = &self.neighbors;
        let mut cand = beam_search(entry, &v, self.ef_build, &self.store, |cur, cb| {
            for &nb in &neighbors_ref[cur as usize] {
                if nb != new_id { cb(nb); }
            }
        });
        cand.retain(|(id, _)| *id != new_id);
        self.rng_prune(&mut cand, self.m);
        for &(nb, _) in &cand {
            self.neighbors[new_id as usize].push(nb);
        }
        for &(nb, _) in &cand {
            let ns = nb as usize;
            self.neighbors[ns].push(new_id);
            if self.neighbors[ns].len() > self.m {
                let mut list: Vec<(NodeId, f32)> = self.neighbors[ns]
                    .iter()
                    .map(|&x| (x, l2_sq(self.store.get(nb), self.store.get(x))))
                    .collect();
                self.rng_prune(&mut list, self.m);
                self.neighbors[ns] = list.into_iter().map(|(id, _)| id).collect();
            }
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
