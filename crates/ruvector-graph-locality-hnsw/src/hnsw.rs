//! Minimal single-file HNSW implementation.
//!
//! Small, dependency-light, correctness-oriented. Not a competitor to hnswlib,
//! but sufficient for A/B experiments on storage layout: identical construction
//! for all reordering variants, differing only in *physical* vector order.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;

/// HNSW hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub m: usize,           // max neighbors per node (non-zero layer)
    pub m0: usize,          // max neighbors at layer 0
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 100,
            seed: 0xC0FFEE_u64,
        }
    }
}

/// The index. `vectors` and `layers` are keyed by **logical** id (insert order).
#[derive(Debug, Clone)]
pub struct HnswIndex {
    pub cfg: HnswConfig,
    pub dim: usize,
    pub vectors: Vec<f32>,        // logical layout: id * dim .. (id+1)*dim
    pub layers: Vec<Vec<Vec<u32>>>, // layers[level][node_id] = neighbors
    pub node_max_layer: Vec<u8>,
    pub entry_point: Option<u32>,
    pub top_layer: usize,
    pub(crate) ml: f32,
    pub(crate) rng: StdRng,
}

impl HnswIndex {
    pub fn new(dim: usize, cfg: HnswConfig) -> Self {
        let ml = 1.0 / (cfg.m as f32).ln();
        Self {
            cfg,
            dim,
            vectors: Vec::new(),
            layers: vec![Vec::new()], // layer 0 always present
            node_max_layer: Vec::new(),
            entry_point: None,
            top_layer: 0,
            ml,
            rng: StdRng::seed_from_u64(cfg.seed),
        }
    }

    pub fn len(&self) -> usize {
        self.node_max_layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, id: u32) -> &[f32] {
        let i = id as usize;
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    fn dist(&self, id: u32, q: &[f32]) -> f32 {
        crate::sq_l2(self.get(id), q)
    }

    fn assign_level(&mut self) -> u8 {
        let r: f32 = self.rng.gen();
        let lvl = (-r.ln() * self.ml).floor() as usize;
        lvl.min(32) as u8
    }

    /// Insert a vector, returns its logical id.
    pub fn insert(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.len() as u32;
        self.vectors.extend_from_slice(v);
        let level = self.assign_level();
        self.node_max_layer.push(level);
        while self.layers.len() <= level as usize {
            self.layers.push(Vec::new());
        }
        for l in 0..=level as usize {
            self.layers[l].push(Vec::new());
        }
        // fill missing rows for layer 0 for previously-inserted-only-at-0 nodes:
        // (all nodes exist at layer 0, so layers[0] length == id+1 already if we pushed above.
        // But for layers > 0 we only push for nodes reaching that level. Instead maintain
        // sparse adjacency using per-level HashMap? For simplicity: pad all layers below
        // `level` to include the current node. Above `level`, adjacency isn't consulted.

        // Actually with the above logic, layers[l] index != node id when l > 0.
        // Rework: store adjacency as Vec keyed by node id, one entry per layer up to `top_layer`.
        // Undo push:
        for l in 0..=level as usize {
            self.layers[l].pop();
        }
        // grow every existing layer to hold node id
        for l in 0..self.layers.len() {
            while self.layers[l].len() <= id as usize {
                self.layers[l].push(Vec::new());
            }
        }
        // ensure top layer covers our level
        while self.layers.len() <= level as usize {
            self.layers.push(Vec::new());
            let need = id as usize + 1;
            let last = self.layers.len() - 1;
            while self.layers[last].len() < need {
                self.layers[last].push(Vec::new());
            }
        }

        // If this is the first node, set as entry point.
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.top_layer = level as usize;
            return id;
        }

        let mut ep = self.entry_point.unwrap();
        let top = self.top_layer;

        // Greedy search from top layer down to level+1
        for l in (level as usize + 1..=top).rev() {
            ep = self.greedy_search_layer(v, ep, l);
        }

        // For each layer from min(level, top) down to 0: search then connect
        let start_layer = (level as usize).min(top);
        let mut cur_ep = ep;
        for l in (0..=start_layer).rev() {
            let candidates = self.search_layer(v, cur_ep, self.cfg.ef_construction, l);
            let max = if l == 0 { self.cfg.m0 } else { self.cfg.m };
            let neighbors = self.select_neighbors_simple(candidates.clone(), max);
            // connect id -> neighbors
            self.layers[l][id as usize] = neighbors.iter().map(|&(_, n)| n).collect();
            // reciprocal, with pruning
            for &(_, n) in &neighbors {
                self.layers[l][n as usize].push(id);
                let overflow = self.layers[l][n as usize].len() > max;
                if overflow {
                    let nb_ids: Vec<u32> = self.layers[l][n as usize].clone();
                    let full: Vec<(f32, u32)> = nb_ids
                        .iter()
                        .map(|&x| (self.dist_between(n, x), x))
                        .collect();
                    let pruned = self.select_neighbors_simple(full, max);
                    self.layers[l][n as usize] = pruned.into_iter().map(|(_, x)| x).collect();
                }
            }
            if !candidates.is_empty() {
                cur_ep = candidates[0].1;
            }
        }

        if level as usize > top {
            self.entry_point = Some(id);
            self.top_layer = level as usize;
        }
        id
    }

    fn dist_between(&self, a: u32, b: u32) -> f32 {
        crate::sq_l2(self.get(a), self.get(b))
    }

    fn greedy_search_layer(&self, q: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_d = crate::sq_l2(self.get(entry), q);
        loop {
            let mut best = current;
            let mut best_d = current_d;
            for &n in &self.layers[layer][current as usize] {
                let d = crate::sq_l2(self.get(n), q);
                if d < best_d {
                    best_d = d;
                    best = n;
                }
            }
            if best == current {
                return current;
            }
            current = best;
            current_d = best_d;
        }
    }

    /// Best-first search at a single layer returning up to `ef` closest by ascending distance.
    pub fn search_layer(&self, q: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<(f32, u32)> {
        // Uses a min-heap of frontier (negated for BinaryHeap) and max-heap of best-so-far.
        use std::cmp::Reverse;
        let mut visited = vec![false; self.len()];
        let entry_d = crate::sq_l2(self.get(entry), q);
        visited[entry as usize] = true;
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut best: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        frontier.push(Reverse((OrdF32(entry_d), entry)));
        best.push((OrdF32(entry_d), entry));
        while let Some(Reverse((OrdF32(d), node))) = frontier.pop() {
            let worst_best = best.peek().map(|x| x.0 .0).unwrap_or(f32::MAX);
            if d > worst_best && best.len() >= ef {
                break;
            }
            for &nb in &self.layers[layer][node as usize] {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let dd = crate::sq_l2(self.get(nb), q);
                let worst = best.peek().map(|x| x.0 .0).unwrap_or(f32::MAX);
                if best.len() < ef || dd < worst {
                    frontier.push(Reverse((OrdF32(dd), nb)));
                    best.push((OrdF32(dd), nb));
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = best.into_iter().map(|(d, n)| (d.0, n)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    fn select_neighbors_simple(&self, mut cands: Vec<(f32, u32)>, m: usize) -> Vec<(f32, u32)> {
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.truncate(m);
        cands
    }
}

/// f32 wrapper that implements `Ord` (panics on NaN).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).expect("NaN in distance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn insert_and_search_returns_self() {
        let mut idx = HnswIndex::new(8, HnswConfig::default());
        let vs = rand_vecs(64, 8, 1);
        for v in &vs {
            idx.insert(v);
        }
        // Query with an inserted vector; expect it to be the nearest.
        let r = idx.search_layer(&vs[7], idx.entry_point.unwrap(), 10, 0);
        assert!(r.iter().any(|&(_, id)| id == 7));
    }
}
