//! Self-contained HNSW implementation with pluggable beam termination.
//!
//! Distance: squared Euclidean (monotone in L2, cheaper).
//!
//! This implementation is intentionally simple and single-threaded. It is
//! a research vehicle for `BeamTerminator`, not a production index.

use crate::terminator::{BeamTerminator, TerminationStats};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub level_lambda: f64,
    pub seed: u64,
}
impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            level_lambda: 1.0 / (16f64).ln(),
            seed: 0xC0FFEE,
        }
    }
}

/// A (id, distance) pair returned from search.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

// --- Ordered helpers for heaps ---------------------------------------------

#[derive(Copy, Clone, Debug)]
struct Cand {
    dist: f32,
    id: u32,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.id == o.id
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
// Min-heap by dist (BinaryHeap is max-heap, so reverse).
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&o.id))
    }
}

#[derive(Copy, Clone, Debug)]
struct MaxCand {
    dist: f32,
    id: u32,
}
impl PartialEq for MaxCand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.id == o.id
    }
}
impl Eq for MaxCand {}
impl PartialOrd for MaxCand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxCand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&o.id))
    }
}

// --- The index --------------------------------------------------------------

pub struct Hnsw {
    dim: usize,
    params: HnswParams,
    data: Vec<f32>,          // flat: id * dim .. (id+1)*dim
    levels: Vec<u8>,         // per-node level
    neighbors: Vec<Vec<Vec<u32>>>, // node -> level -> neighbours
    entry: Option<u32>,
    rng: ChaCha8Rng,
}

#[inline]
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

impl Hnsw {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(params.seed);
        Self {
            dim,
            params,
            data: Vec::new(),
            levels: Vec::new(),
            neighbors: Vec::new(),
            entry: None,
            rng,
        }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn vec_at(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    fn random_level(&mut self) -> u8 {
        let r: f64 = self.rng.gen_range(1e-12..1.0);
        let lvl = (-r.ln() * self.params.level_lambda).floor();
        lvl.max(0.0).min(16.0) as u8
    }

    pub fn insert(&mut self, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim);
        let id = self.levels.len() as u32;
        let lvl = self.random_level();
        self.data.extend_from_slice(vec);
        self.levels.push(lvl);
        self.neighbors
            .push((0..=lvl as usize).map(|_| Vec::new()).collect());

        let entry = match self.entry {
            None => {
                self.entry = Some(id);
                return;
            }
            Some(e) => e,
        };

        // Find entry by greedy descent down levels above `lvl`.
        let top_level = self.levels[entry as usize];
        let mut curr = entry;
        let mut curr_d = sq_l2(self.vec_at(curr), vec);
        let mut l = top_level as i32;
        while l > lvl as i32 {
            let improved = self.greedy_layer(vec, &mut curr, &mut curr_d, l as usize);
            if !improved {
                // continue descent anyway
            }
            l -= 1;
        }

        // From level=min(top,lvl) down to 0, search ef_construction and link.
        let start_lvl = (top_level.min(lvl)) as i32;
        for layer in (0..=start_lvl).rev() {
            let candidates = self.search_layer_construct(vec, curr, layer as usize);
            // Pick M neighbours.
            let mmax = if layer == 0 { self.params.m0 } else { self.params.m };
            let mut selected = self.select_neighbors(vec, &candidates, mmax);
            // Update curr for next descent.
            if let Some(best) = candidates.first() {
                curr = best.id;
            }
            // Bidirectional links.
            for n in &selected {
                self.neighbors[id as usize][layer as usize].push(n.id);
            }
            let selected_ids: Vec<u32> = selected.drain(..).map(|n| n.id).collect();
            for nid in selected_ids {
                self.neighbors[nid as usize][layer as usize].push(id);
                // Trim neighbour's list if it overflows mmax.
                let nlist = &self.neighbors[nid as usize][layer as usize];
                if nlist.len() > mmax {
                    let n_vec_owned = self.vec_at(nid).to_vec();
                    let cands: Vec<Cand> = nlist
                        .iter()
                        .map(|&x| Cand {
                            id: x,
                            dist: sq_l2(self.vec_at(x), &n_vec_owned),
                        })
                        .collect();
                    let mut sorted = cands;
                    sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
                    sorted.truncate(mmax);
                    let trimmed = sorted.iter().map(|c| c.id).collect::<Vec<_>>();
                    self.neighbors[nid as usize][layer as usize] = trimmed;
                }
            }
        }

        // Possibly raise entry point.
        if lvl > self.levels[self.entry.unwrap() as usize] {
            self.entry = Some(id);
        }
    }

    fn greedy_layer(&self, q: &[f32], curr: &mut u32, curr_d: &mut f32, layer: usize) -> bool {
        let mut improved = false;
        loop {
            let nbrs = &self.neighbors[*curr as usize][layer];
            let mut best = *curr;
            let mut best_d = *curr_d;
            for &nb in nbrs {
                let d = sq_l2(self.vec_at(nb), q);
                if d < best_d {
                    best = nb;
                    best_d = d;
                }
            }
            if best == *curr {
                return improved;
            }
            *curr = best;
            *curr_d = best_d;
            improved = true;
        }
    }

    fn search_layer_construct(&self, q: &[f32], entry: u32, layer: usize) -> Vec<Cand> {
        let ef = self.params.ef_construction;
        let mut visited = HashSet::with_capacity(ef * 4);
        let mut cand: BinaryHeap<Cand> = BinaryHeap::new();
        let mut res: BinaryHeap<MaxCand> = BinaryHeap::new();
        let d0 = sq_l2(self.vec_at(entry), q);
        cand.push(Cand { id: entry, dist: d0 });
        res.push(MaxCand { id: entry, dist: d0 });
        visited.insert(entry);
        while let Some(top) = cand.pop() {
            let worst = res.peek().map(|m| m.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst {
                break;
            }
            for &nb in &self.neighbors[top.id as usize][layer] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = sq_l2(self.vec_at(nb), q);
                if res.len() < ef || d < res.peek().unwrap().dist {
                    cand.push(Cand { id: nb, dist: d });
                    res.push(MaxCand { id: nb, dist: d });
                    if res.len() > ef {
                        res.pop();
                    }
                }
            }
        }
        let mut v: Vec<Cand> = res
            .into_iter()
            .map(|m| Cand {
                id: m.id,
                dist: m.dist,
            })
            .collect();
        v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        v
    }

    /// Simple neighbour-selection heuristic: take the m nearest.
    fn select_neighbors(&self, _q: &[f32], cands: &[Cand], m: usize) -> Vec<Cand> {
        let mut v = cands.to_vec();
        v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        v.truncate(m);
        v
    }

    // -- Search with pluggable terminator ----------------------------------

    /// Search returning top-k with a custom terminator at layer 0. Upper
    /// layers always use greedy descent.
    pub fn search<T: BeamTerminator + ?Sized>(
        &self,
        q: &[f32],
        k: usize,
        terminator: &mut T,
    ) -> (Vec<Neighbor>, TerminationStats) {
        terminator.reset();
        let mut stats = TerminationStats::default();
        let entry = match self.entry {
            None => return (vec![], stats),
            Some(e) => e,
        };

        // Greedy descent above layer 0.
        let mut curr = entry;
        let mut curr_d = sq_l2(self.vec_at(curr), q);
        stats.distance_evals += 1;
        let top_level = self.levels[entry as usize];
        for layer in (1..=top_level as usize).rev() {
            // greedy_layer mutates curr/curr_d via &mut.
            let _ = self.greedy_layer_with_stats(q, &mut curr, &mut curr_d, layer, &mut stats);
        }

        // Layer-0 beam search with terminator. The result pool holds up to
        // `ef` candidates (classical HNSW). The final top-k is truncated at
        // return time.
        let ef = terminator.ef_pool().max(k);
        let mut visited = HashSet::with_capacity(ef * 4);
        let mut cand: BinaryHeap<Cand> = BinaryHeap::new();
        let mut res: BinaryHeap<MaxCand> = BinaryHeap::new();
        // Parallel top-k view: max-heap of size <= k. Used to feed the
        // adaptive terminator with `worst_topk` (kth-best distance), which
        // is a much tighter signal than worst-of-ef.
        let mut topk_view: BinaryHeap<MaxCand> = BinaryHeap::new();
        let push_topk = |topk_view: &mut BinaryHeap<MaxCand>, k: usize, m: MaxCand| {
            if topk_view.len() < k {
                topk_view.push(m);
            } else if m.dist < topk_view.peek().unwrap().dist {
                topk_view.pop();
                topk_view.push(m);
            }
        };

        cand.push(Cand {
            id: curr,
            dist: curr_d,
        });
        res.push(MaxCand { id: curr, dist: curr_d });
        push_topk(&mut topk_view, k, MaxCand { id: curr, dist: curr_d });
        visited.insert(curr);

        loop {
            let top = match cand.pop() {
                Some(t) => t,
                None => break,
            };
            let worst_ef = res.peek().map(|m| m.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst_ef && res.len() >= ef {
                // Classical safety stop.
                break;
            }
            stats.expansions += 1;
            let mut improved_this_step = false;
            let worst_topk_before = topk_view
                .peek()
                .map(|m| m.dist)
                .unwrap_or(f32::INFINITY);
            for &nb in &self.neighbors[top.id as usize][0] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = sq_l2(self.vec_at(nb), q);
                stats.distance_evals += 1;
                if res.len() < ef || d < res.peek().unwrap().dist {
                    cand.push(Cand { id: nb, dist: d });
                    res.push(MaxCand { id: nb, dist: d });
                    if res.len() > ef {
                        res.pop();
                    }
                    push_topk(&mut topk_view, k, MaxCand { id: nb, dist: d });
                    improved_this_step = true;
                }
            }
            let worst_topk_after = topk_view
                .peek()
                .map(|m| m.dist)
                .unwrap_or(f32::INFINITY);
            if worst_topk_after < worst_topk_before {
                stats.improvements += 1;
            }
            let min_unexpanded = cand.peek().map(|c| c.dist).unwrap_or(f32::INFINITY);
            if terminator.should_stop(
                stats.expansions,
                min_unexpanded,
                worst_topk_after,
                improved_this_step,
            ) {
                stats.early_stopped = true;
                break;
            }
        }

        let mut out: Vec<Neighbor> = res
            .into_iter()
            .map(|m| Neighbor { id: m.id, dist: m.dist })
            .collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out.truncate(k);
        (out, stats)
    }

    fn greedy_layer_with_stats(
        &self,
        q: &[f32],
        curr: &mut u32,
        curr_d: &mut f32,
        layer: usize,
        stats: &mut TerminationStats,
    ) -> bool {
        let mut improved = false;
        loop {
            let nbrs = &self.neighbors[*curr as usize][layer];
            let mut best = *curr;
            let mut best_d = *curr_d;
            for &nb in nbrs {
                let d = sq_l2(self.vec_at(nb), q);
                stats.distance_evals += 1;
                if d < best_d {
                    best = nb;
                    best_d = d;
                }
            }
            if best == *curr {
                return improved;
            }
            *curr = best;
            *curr_d = best_d;
            improved = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminator::FixedEfTerminator;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn brute_force(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
        let mut v: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, x)| (i as u32, sq_l2(x, q)))
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v.iter().take(k).map(|(i, _)| *i).collect()
    }

    #[test]
    fn hnsw_recall_at_10_above_85_for_small_dataset() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let dim = 32;
        let n = 1000;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let mut idx = Hnsw::new(dim, HnswParams::default());
        for v in &data {
            idx.insert(v);
        }
        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth = brute_force(&data, q, 10);
            let mut term = FixedEfTerminator::new(64);
            let (res, _) = idx.search(q, 10, &mut term);
            let res_ids: std::collections::HashSet<u32> = res.iter().map(|n| n.id).collect();
            for t in truth {
                total += 1;
                if res_ids.contains(&t) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / total as f64;
        assert!(recall > 0.80, "recall@10 too low: {recall}");
    }
}
