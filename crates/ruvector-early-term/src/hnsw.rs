//! Minimal HNSW with a pluggable termination policy.
//!
//! Layered proximity graph with hierarchical entry. We only implement what is
//! needed to study early-termination tradeoffs: build via greedy insertion,
//! search via beam expansion under a `TerminationPolicy`.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::data::{cos_dist, dot};
use crate::policy::TerminationPolicy;

#[derive(Clone, Copy, Debug)]
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
            ml: 1.0 / std::f32::consts::LN_2 / 4.0,
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchStats {
    pub distance_calls: u64,
    pub expansions: u64,
    pub terminated_early: bool,
    pub stop_step: u32,
}

struct Cand {
    dist: f32,
    id: u32,
}
impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for Cand {}
impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}

// Max-heap by dist (for keeping the k smallest)
struct CandMax(Cand);
impl PartialEq for CandMax { fn eq(&self, o: &Self) -> bool { self.0.dist == o.0.dist } }
impl Eq for CandMax {}
impl PartialOrd for CandMax { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for CandMax {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.dist.partial_cmp(&o.0.dist).unwrap_or(Ordering::Equal)
    }
}
// Min-heap by dist
struct CandMin(Cand);
impl PartialEq for CandMin { fn eq(&self, o: &Self) -> bool { self.0.dist == o.0.dist } }
impl Eq for CandMin {}
impl PartialOrd for CandMin { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for CandMin {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}

pub struct Hnsw {
    pub params: HnswParams,
    pub dim: usize,
    vectors: Vec<Vec<f32>>,
    // levels[id] = max level for node id (0..=L)
    levels: Vec<u8>,
    // links[level][id] = neighbor list
    links: Vec<Vec<Vec<u32>>>,
    entry: u32,
    top_level: u8,
}

impl Hnsw {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        Self {
            params,
            dim,
            vectors: Vec::new(),
            levels: Vec::new(),
            links: vec![Vec::new()],
            entry: 0,
            top_level: 0,
        }
    }

    #[inline]
    fn d(&self, a: u32, q: &[f32]) -> f32 {
        cos_dist(q, &self.vectors[a as usize])
    }

    pub fn build(&mut self, vecs: Vec<Vec<f32>>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.params.seed);
        let n = vecs.len();
        self.vectors = vecs;
        self.levels = Vec::with_capacity(n);
        // Determine levels first
        for _ in 0..n {
            let r: f32 = rng.gen_range(1e-9..1.0);
            let lvl = (-(r.ln()) * self.params.ml).floor() as u32;
            let lvl = lvl.min(8) as u8;
            self.levels.push(lvl);
            if (lvl as usize) >= self.links.len() {
                while self.links.len() <= lvl as usize {
                    self.links.push(Vec::new());
                }
            }
        }
        // Ensure link tables sized
        for lvl in 0..self.links.len() {
            self.links[lvl] = vec![Vec::new(); n];
        }

        // Insert in order
        self.entry = 0;
        self.top_level = self.levels[0];

        for id in 0..n {
            if id == 0 { continue; }
            let id_u = id as u32;
            let id_level = self.levels[id];
            let q = self.vectors[id].clone();

            // Greedy descent from top_level to id_level+1
            let mut cur = self.entry;
            let mut cur_d = self.d(cur, &q);
            if self.top_level > id_level {
                for lvl in (id_level as usize + 1..=self.top_level as usize).rev() {
                    let mut changed = true;
                    while changed {
                        changed = false;
                        let neigh = self.links[lvl][cur as usize].clone();
                        for &n in &neigh {
                            let nd = self.d(n, &q);
                            if nd < cur_d { cur = n; cur_d = nd; changed = true; }
                        }
                    }
                }
            }

            // Beam-search insert at each level from min(top_level,id_level) down to 0
            let start_lvl = self.top_level.min(id_level) as i32;
            let mut ep_set = vec![(cur_d, cur)];
            for lvl in (0..=start_lvl).rev() {
                let lvl_u = lvl as usize;
                let cands = self.search_layer_build(&q, &ep_set, lvl_u, self.params.ef_construction);
                let m_max = if lvl_u == 0 { self.params.m_max0 } else { self.params.m };
                let selected = self.select_neighbors_heuristic(&q, &cands, self.params.m);
                self.links[lvl_u][id] = selected.iter().map(|c| c.1).collect();
                // Add back-edges
                for &(_d, nb) in &selected {
                    let nbu = nb as usize;
                    let mut nb_links = self.links[lvl_u][nbu].clone();
                    nb_links.push(id_u);
                    if nb_links.len() > m_max {
                        let mut nb_cands: Vec<(f32, u32)> = nb_links.iter()
                            .map(|&x| (self.d(x, &self.vectors[nbu].clone()), x))
                            .collect();
                        nb_cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        nb_links = nb_cands.into_iter().take(m_max).map(|c| c.1).collect();
                    }
                    self.links[lvl_u][nbu] = nb_links;
                }
                ep_set = cands.into_iter().take(self.params.ef_construction.min(8)).collect();
            }

            if id_level > self.top_level {
                self.top_level = id_level;
                self.entry = id_u;
            }
        }
    }

    fn select_neighbors_heuristic(&self, _q: &[f32], cands: &[(f32, u32)], m: usize) -> Vec<(f32, u32)> {
        // Simple greedy: keep closest m. (Heuristic per Malkov skipped to keep crate compact.)
        let mut out = cands.to_vec();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.truncate(m);
        out
    }

    fn search_layer_build(
        &self,
        q: &[f32],
        ep: &[(f32, u32)],
        level: usize,
        ef: usize,
    ) -> Vec<(f32, u32)> {
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<CandMin> = BinaryHeap::new();
        let mut top: BinaryHeap<CandMax> = BinaryHeap::new();
        for &(d, id) in ep {
            visited[id as usize] = true;
            candidates.push(CandMin(Cand { dist: d, id }));
            top.push(CandMax(Cand { dist: d, id }));
        }
        while let Some(CandMin(c)) = candidates.pop() {
            let upper = top.peek().map(|x| x.0.dist).unwrap_or(f32::INFINITY);
            if c.dist > upper { break; }
            for &nb in &self.links[level][c.id as usize] {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let nd = self.d(nb, q);
                let upper = top.peek().map(|x| x.0.dist).unwrap_or(f32::INFINITY);
                if top.len() < ef || nd < upper {
                    candidates.push(CandMin(Cand { dist: nd, id: nb }));
                    top.push(CandMax(Cand { dist: nd, id: nb }));
                    if top.len() > ef { top.pop(); }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = top.into_iter().map(|c| (c.0.dist, c.0.id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    /// Search with a pluggable termination policy at the bottom level.
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        ef_max: usize,
        policy: &mut dyn TerminationPolicy,
    ) -> (Vec<(f32, u32)>, SearchStats) {
        let mut stats = SearchStats::default();
        // Greedy descent through upper levels.
        let mut cur = self.entry;
        let mut cur_d = cos_dist(q, &self.vectors[cur as usize]);
        stats.distance_calls += 1;
        for lvl in (1..=self.top_level as usize).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &nb in &self.links[lvl][cur as usize] {
                    let nd = cos_dist(q, &self.vectors[nb as usize]);
                    stats.distance_calls += 1;
                    if nd < cur_d {
                        cur = nb;
                        cur_d = nd;
                        changed = true;
                    }
                }
            }
        }

        // Bottom-level beam search with termination policy.
        policy.reset(k, ef_max);
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<CandMin> = BinaryHeap::new();
        let mut top: BinaryHeap<CandMax> = BinaryHeap::new();
        visited[cur as usize] = true;
        candidates.push(CandMin(Cand { dist: cur_d, id: cur }));
        top.push(CandMax(Cand { dist: cur_d, id: cur }));

        let mut step: u32 = 0;
        while let Some(CandMin(c)) = candidates.pop() {
            step += 1;
            stats.expansions += 1;
            let upper = top.peek().map(|x| x.0.dist).unwrap_or(f32::INFINITY);
            if c.dist > upper && top.len() >= ef_max {
                break;
            }

            // Termination check BEFORE expanding new node.
            let kth = if top.len() >= k {
                // peek-min of top-k: requires scanning, k is small (10).
                let mut all: Vec<f32> = top.iter().map(|x| x.0.dist).collect();
                all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                all[k - 1]
            } else {
                f32::INFINITY
            };
            if policy.should_stop(step, c.dist, kth, top.len()) {
                stats.terminated_early = true;
                stats.stop_step = step;
                break;
            }

            for &nb in &self.links[0][c.id as usize] {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let nd = cos_dist(q, &self.vectors[nb as usize]);
                stats.distance_calls += 1;
                let upper = top.peek().map(|x| x.0.dist).unwrap_or(f32::INFINITY);
                if top.len() < ef_max || nd < upper {
                    candidates.push(CandMin(Cand { dist: nd, id: nb }));
                    top.push(CandMax(Cand { dist: nd, id: nb }));
                    if top.len() > ef_max { top.pop(); }
                }
            }
        }

        if !stats.terminated_early {
            stats.stop_step = step;
        }

        let mut out: Vec<(f32, u32)> = top.into_iter().map(|c| (c.0.dist, c.0.id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.truncate(k);
        (out, stats)
    }
}

// Suppress unused-import warning at compile when dot only used in data.rs.
#[allow(dead_code)]
fn _force_dot(a: &[f32], b: &[f32]) -> f32 { dot(a, b) }
