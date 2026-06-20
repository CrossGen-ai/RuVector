//! Tiny pure-Rust HNSW (L2-squared, in-memory). Sufficient for
//! evaluating early-termination policies; not optimised for SIMD.
//!
//! Public surface:
//! * [`HnswParams`] — index parameters.
//! * [`Hnsw`] — index struct with `insert` and `search` methods.
//! * [`SearchTrace`] — records the per-step best-distance trajectory
//!   used by the LAET predictor.
//!
//! All randomness is seeded so benchmarks are reproducible.

use crate::data::l2sq;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Debug)]
pub struct HnswParams {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ml: f32,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ml: 1.0 / (16f32).ln(),
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchTrace {
    pub dist_calls: u32,
    /// Best distance after each candidate pop. The series is
    /// monotonically non-increasing.
    pub best_progress: Vec<f32>,
    /// Number of "no-improvement" pops in a row at termination.
    pub stalled_steps: u32,
}

// Min-heap by distance for candidates; max-heap by distance for the
// dynamic result set. Implemented with wrappers around f32 ordered by
// total_cmp so NaN never crashes.

#[derive(Copy, Clone, Debug)]
struct DistNode {
    dist: f32,
    id: u32,
}
impl PartialEq for DistNode {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.id == o.id
    }
}
impl Eq for DistNode {}
impl Ord for DistNode {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist)
    }
}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Copy, Clone, Debug)]
struct MaxDist(DistNode);
impl PartialEq for MaxDist {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MaxDist {}
impl Ord for MaxDist {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.dist.total_cmp(&o.0.dist)
    }
}
impl PartialOrd for MaxDist {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Copy, Clone, Debug)]
struct MinDist(DistNode);
impl PartialEq for MinDist {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MinDist {}
impl Ord for MinDist {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.dist.total_cmp(&self.0.dist)
    }
}
impl PartialOrd for MinDist {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

pub struct Hnsw {
    pub dim: usize,
    pub params: HnswParams,
    pub data: Vec<Vec<f32>>,
    /// `layers[l][node] = neighbours`
    layers: Vec<Vec<Vec<u32>>>,
    /// per-node assigned top layer
    node_layer: Vec<u8>,
    entry: Option<u32>,
    top_layer: u8,
    rng: ChaCha8Rng,
}

impl Hnsw {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(params.seed);
        Self {
            dim,
            params,
            data: Vec::new(),
            layers: vec![Vec::new()],
            node_layer: Vec::new(),
            entry: None,
            top_layer: 0,
            rng,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn random_level(&mut self) -> u8 {
        let r: f32 = self.rng.gen_range(1e-9f32..1.0);
        (-(r.ln()) * self.params.ml).floor() as u8
    }

    pub fn insert(&mut self, v: Vec<f32>) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.data.len() as u32;
        let l = self.random_level();

        // Grow layer stacks. New layers must be back-filled with
        // empty neighbour slots for every previously-inserted node so
        // that `self.layers[L][node_id]` is always a valid index for
        // any `L < self.layers.len()` and `node_id < self.data.len()`.
        let prev_n = self.data.len();
        while (self.layers.len() as u8) <= l {
            let mut new_layer = Vec::with_capacity(prev_n + 1);
            for _ in 0..prev_n {
                new_layer.push(Vec::new());
            }
            self.layers.push(new_layer);
        }
        for layer in self.layers.iter_mut() {
            layer.push(Vec::new());
        }
        self.data.push(v);
        self.node_layer.push(l);

        if self.entry.is_none() {
            self.entry = Some(id);
            self.top_layer = l;
            return id;
        }

        let q = self.data[id as usize].clone();
        let mut ep = self.entry.unwrap();
        let mut cur_dist = l2sq(&q, &self.data[ep as usize]);

        // Greedy descend from top to l+1
        for lc in (l + 1..=self.top_layer).rev() {
            loop {
                let mut changed = false;
                let nbrs = self.layers[lc as usize][ep as usize].clone();
                for n in nbrs {
                    let d = l2sq(&q, &self.data[n as usize]);
                    if d < cur_dist {
                        cur_dist = d;
                        ep = n;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Bottom-up insertion at layers l..=0
        for lc in (0..=l.min(self.top_layer)).rev() {
            let ef = self.params.ef_construction;
            let (candidates, _trace) = self.search_layer(&q, ep, ef, lc, false);
            let m_max = if lc == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            let selected = self.select_neighbours(&q, &candidates, m_max);
            // bidirectional links
            self.layers[lc as usize][id as usize] = selected.clone();
            for &n in &selected {
                let nn = &mut self.layers[lc as usize][n as usize];
                nn.push(id);
                if nn.len() > m_max {
                    // shrink: keep closest m_max
                    let n_vec = self.data[n as usize].clone();
                    let mut pool: Vec<u32> = nn.clone();
                    let chosen = self.select_neighbours_for_node(&n_vec, &pool, m_max);
                    pool.clear();
                    pool.extend(chosen);
                    self.layers[lc as usize][n as usize] = pool;
                }
            }
            if let Some(&first) = selected.first() {
                ep = first;
            }
        }

        if l > self.top_layer {
            self.top_layer = l;
            self.entry = Some(id);
        }
        id
    }

    /// Diverse-neighbour heuristic (Malkov & Yashunin, Algorithm 4):
    /// admit a candidate only if it is closer to the pivot `q` than
    /// to any already-selected neighbour. This avoids the path
    /// collapse that plain "closest-m" selection produces and is
    /// what gives canonical HNSW its high recall.
    fn select_neighbours(&self, q: &[f32], cands: &[u32], m: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = cands
            .iter()
            .map(|&c| (l2sq(q, &self.data[c as usize]), c))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut chosen: Vec<u32> = Vec::with_capacity(m);
        for (d_q, c) in scored {
            if chosen.len() >= m {
                break;
            }
            let mut keep = true;
            for &x in &chosen {
                let d_xc = l2sq(&self.data[x as usize], &self.data[c as usize]);
                if d_xc < d_q {
                    keep = false;
                    break;
                }
            }
            if keep {
                chosen.push(c);
            }
        }
        chosen
    }

    fn select_neighbours_for_node(&self, n_vec: &[f32], cands: &[u32], m: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = cands
            .iter()
            .map(|&c| (l2sq(n_vec, &self.data[c as usize]), c))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut chosen: Vec<u32> = Vec::with_capacity(m);
        for (d_q, c) in scored {
            if chosen.len() >= m {
                break;
            }
            let mut keep = true;
            for &x in &chosen {
                let d_xc = l2sq(&self.data[x as usize], &self.data[c as usize]);
                if d_xc < d_q {
                    keep = false;
                    break;
                }
            }
            if keep {
                chosen.push(c);
            }
        }
        chosen
    }

    /// Layer-local search returning candidate ids + diagnostic trace.
    fn search_layer(
        &self,
        q: &[f32],
        ep: u32,
        ef: usize,
        layer: u8,
        record: bool,
    ) -> (Vec<u32>, SearchTrace) {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut cand: BinaryHeap<MinDist> = BinaryHeap::new();
        let mut top: BinaryHeap<MaxDist> = BinaryHeap::new();
        let mut trace = SearchTrace::default();

        let d0 = l2sq(q, &self.data[ep as usize]);
        trace.dist_calls += 1;
        cand.push(MinDist(DistNode { dist: d0, id: ep }));
        top.push(MaxDist(DistNode { dist: d0, id: ep }));
        visited.insert(ep);

        let mut best = d0;
        let mut stalled: u32 = 0;
        if record {
            trace.best_progress.push(best);
        }

        while let Some(MinDist(c)) = cand.pop() {
            let worst_top = top.peek().map(|t| t.0.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst_top && top.len() >= ef {
                break;
            }
            let nbrs = &self.layers[layer as usize][c.id as usize];
            let mut improved = false;
            for &n in nbrs {
                if !visited.insert(n) {
                    continue;
                }
                let d = l2sq(q, &self.data[n as usize]);
                trace.dist_calls += 1;
                let worst = top.peek().map(|t| t.0.dist).unwrap_or(f32::INFINITY);
                if d < worst || top.len() < ef {
                    cand.push(MinDist(DistNode { dist: d, id: n }));
                    top.push(MaxDist(DistNode { dist: d, id: n }));
                    if top.len() > ef {
                        top.pop();
                    }
                    if d < best {
                        best = d;
                        improved = true;
                    }
                }
            }
            if record {
                trace.best_progress.push(best);
            }
            if improved {
                stalled = 0;
            } else {
                stalled = stalled.saturating_add(1);
            }
        }
        trace.stalled_steps = stalled;

        let mut out: Vec<u32> = top.into_iter().map(|m| m.0.id).collect();
        out.sort_by(|&a, &b| {
            l2sq(q, &self.data[a as usize])
                .total_cmp(&l2sq(q, &self.data[b as usize]))
        });
        (out, trace)
    }

    /// Greedy descent through upper layers down to layer 1, returning
    /// the entry point for layer-0 search and the descent trace.
    pub fn entry_for_layer0(&self, q: &[f32]) -> (u32, SearchTrace) {
        let mut trace = SearchTrace::default();
        let mut ep = match self.entry {
            Some(e) => e,
            None => return (0, trace),
        };
        let mut cur_dist = l2sq(q, &self.data[ep as usize]);
        trace.dist_calls += 1;
        trace.best_progress.push(cur_dist);
        for lc in (1..=self.top_layer).rev() {
            loop {
                let mut changed = false;
                let nbrs = self.layers[lc as usize][ep as usize].clone();
                for n in nbrs {
                    let d = l2sq(q, &self.data[n as usize]);
                    trace.dist_calls += 1;
                    if d < cur_dist {
                        cur_dist = d;
                        ep = n;
                        changed = true;
                    }
                }
                trace.best_progress.push(cur_dist);
                if !changed {
                    break;
                }
            }
        }
        (ep, trace)
    }

    /// Layer-0 search with a fixed `ef`. Returns (top-k ids in distance
    /// order, trace). The trace records the per-pop best-distance
    /// trajectory which feeds the LAET predictor.
    pub fn search_layer0(&self, q: &[f32], ef: usize, k: usize) -> (Vec<u32>, SearchTrace) {
        let (ep, mut t0) = self.entry_for_layer0(q);
        let (mut cands, mut t1) = self.search_layer(q, ep, ef, 0, true);
        t0.dist_calls += t1.dist_calls;
        t0.best_progress.append(&mut t1.best_progress);
        t0.stalled_steps = t1.stalled_steps;
        cands.truncate(k);
        (cands, t0)
    }
}
