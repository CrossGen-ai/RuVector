//! HNSW coarse quantizer.
//!
//! Small, self-contained multi-layer HNSW over the IVF centroid list. Only
//! the pieces needed for the coarse-quantizer role are implemented:
//!   * insertion with layer sampling (level ~ floor(-ln(u) * mL))
//!   * greedy top-layer descent
//!   * best-first `ef` beam search on the bottom layer
//!
//! No deletion, no external persistence, no serialization. Distances are
//! squared L2. See `IvfCoarse::top_lists` for the entry point.

use crate::{sq_l2, CoarseError, IvfCoarse};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for [`HnswCoarse`].
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Max out-degree at layer 0.
    pub m0: usize,
    /// Max out-degree at layer > 0.
    pub m: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Beam width during search.
    pub ef_search: usize,
    /// RNG seed for reproducibility of level sampling and tie-breaking.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        // Modest defaults; benchmark sweeps ef_search.
        Self {
            m0: 32,
            m: 16,
            ef_construction: 100,
            ef_search: 32,
            seed: 0xA17E_C0DE,
        }
    }
}

/// HNSW coarse quantizer over IVF centroids.
#[derive(Debug)]
pub struct HnswCoarse {
    cfg: HnswConfig,
    d: usize,
    centroids: Vec<f32>,
    // Per-node level (number of layers this node participates in - 1).
    levels: Vec<u8>,
    // graph[layer][node] = Vec<u32> neighbors on that layer.
    graph: Vec<Vec<Vec<u32>>>,
    entry_point: u32,
    max_layer: usize,
    ml: f64, // level normalization
}

// Min/max-heap glue over `(f32, u32)`.
#[derive(Copy, Clone, Debug)]
struct Ord32(f32, u32);
impl PartialEq for Ord32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for Ord32 {}
impl Ord for Ord32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for Ord32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswCoarse {
    /// Build with the supplied [`HnswConfig`].
    pub fn with_config(cfg: HnswConfig) -> Self {
        Self {
            cfg,
            d: 0,
            centroids: Vec::new(),
            levels: Vec::new(),
            graph: Vec::new(),
            entry_point: 0,
            max_layer: 0,
            ml: 1.0 / (2f64).ln(),
        }
    }

    /// Convenience constructor with defaults.
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    #[inline]
    fn vec(&self, i: u32) -> &[f32] {
        let off = i as usize * self.d;
        &self.centroids[off..off + self.d]
    }

    fn assign_level(&self, rng: &mut StdRng) -> usize {
        let u: f64 = rng.gen_range(1.0e-12..1.0);
        (-u.ln() * self.ml).floor() as usize
    }

    // Greedy descent on layer `l` from `ep`, returning nearest node found.
    fn greedy(&self, q: &[f32], ep: u32, l: usize) -> u32 {
        let mut cur = ep;
        let mut cur_d = sq_l2(q, self.vec(cur));
        loop {
            let mut improved = false;
            for &nb in &self.graph[l][cur as usize] {
                let d = sq_l2(q, self.vec(nb));
                if d < cur_d {
                    cur_d = d;
                    cur = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        cur
    }

    // Standard HNSW `search_layer` with beam width `ef`.
    // Returns W (top-ef nearest to q) as an ascending Vec<(dist, id)>.
    fn search_layer(&self, q: &[f32], entry: u32, ef: usize, l: usize) -> Vec<(f32, u32)> {
        let d0 = sq_l2(q, self.vec(entry));
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        visited.insert(entry);
        // Candidates: min-heap of nearest yet-to-expand.
        let mut cand: BinaryHeap<std::cmp::Reverse<Ord32>> = BinaryHeap::new();
        cand.push(std::cmp::Reverse(Ord32(d0, entry)));
        // W: max-heap so we can pop the worst kept.
        let mut w: BinaryHeap<Ord32> = BinaryHeap::new();
        w.push(Ord32(d0, entry));

        while let Some(std::cmp::Reverse(Ord32(cd, c))) = cand.pop() {
            let furthest = w.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
            if cd > furthest {
                break;
            }
            for &nb in &self.graph[l][c as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let nd = sq_l2(q, self.vec(nb));
                let furthest = w.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
                if w.len() < ef || nd < furthest {
                    cand.push(std::cmp::Reverse(Ord32(nd, nb)));
                    w.push(Ord32(nd, nb));
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = w.into_iter().map(|Ord32(d, i)| (d, i)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    // Neighbor selection: simple heuristic = keep the m nearest, skip
    // candidates already covered by a closer kept neighbor ("relative
    // neighborhood graph" pruning as in Malkov & Yashunin 2018 §4.).
    fn select_neighbors(&self, q: &[f32], cands: &[(f32, u32)], m: usize) -> Vec<u32> {
        let mut kept: Vec<(f32, u32)> = Vec::with_capacity(m);
        // cands must be ascending in distance.
        for &(d, id) in cands {
            if kept.len() >= m {
                break;
            }
            let mut covered = false;
            for &(_, kid) in &kept {
                if sq_l2(self.vec(id), self.vec(kid)) < d {
                    covered = true;
                    break;
                }
            }
            if !covered {
                kept.push((d, id));
            }
        }
        // If pruning left us short, fill with plain nearest.
        if kept.len() < m {
            for &(d, id) in cands {
                if kept.len() >= m {
                    break;
                }
                if !kept.iter().any(|&(_, k)| k == id) {
                    kept.push((d, id));
                }
            }
        }
        // Ignore q parameter: kept the signature for symmetry with future
        // heuristics that want it.
        let _ = q;
        kept.into_iter().map(|(_, i)| i).collect()
    }

    fn insert(&mut self, id: u32, level: usize) {
        // Ensure graph has enough layers.
        while self.graph.len() <= level {
            self.graph
                .push(vec![Vec::new(); self.centroids.len() / self.d]);
        }
        // Growing per-node vecs on previously existing layers happens
        // implicitly because we pre-sized above -- but only for the *new*
        // layer. Ensure all existing layers also have this node.
        for layer in self.graph.iter_mut() {
            if layer.len() <= id as usize {
                layer.resize(id as usize + 1, Vec::new());
            }
        }
        self.levels.push(level as u8);

        if id == 0 {
            self.entry_point = 0;
            self.max_layer = level;
            return;
        }

        let q = self.vec(id).to_vec();
        // Phase 1: greedy descent from the current entry through layers
        // above `level` (exclusive).
        let mut ep = self.entry_point;
        for l in (level + 1..=self.max_layer).rev() {
            ep = self.greedy(&q, ep, l);
        }
        // Phase 2: on each layer from min(level, max_layer) down to 0,
        // beam search then wire.
        let top = level.min(self.max_layer);
        for l in (0..=top).rev() {
            let cands = self.search_layer(&q, ep, self.cfg.ef_construction, l);
            let m = if l == 0 { self.cfg.m0 } else { self.cfg.m };
            let neighbors = self.select_neighbors(&q, &cands, m);
            // Wire out-edges from `id`.
            self.graph[l][id as usize] = neighbors.clone();
            // Wire back-edges and prune the neighbors if over-degree.
            for &nb in &neighbors {
                self.graph[l][nb as usize].push(id);
                let cap = if l == 0 { self.cfg.m0 } else { self.cfg.m };
                if self.graph[l][nb as usize].len() > cap {
                    // Re-prune nb's edge list.
                    let nb_vec = self.vec(nb).to_vec();
                    let mut scored: Vec<(f32, u32)> = self.graph[l][nb as usize]
                        .iter()
                        .map(|&x| (sq_l2(&nb_vec, self.vec(x)), x))
                        .collect();
                    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    self.graph[l][nb as usize] =
                        self.select_neighbors(&nb_vec, &scored, cap);
                }
            }
            ep = cands.first().map(|&(_, i)| i).unwrap_or(ep);
        }
        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = id;
        }
    }
}

impl Default for HnswCoarse {
    fn default() -> Self {
        Self::new()
    }
}

impl IvfCoarse for HnswCoarse {
    fn fit(&mut self, centroids: &[f32], d: usize) -> Result<(), CoarseError> {
        if d == 0 || centroids.len() % d != 0 {
            return Err(CoarseError::DimensionMismatch {
                expected: d,
                got: centroids.len(),
            });
        }
        self.d = d;
        self.centroids = centroids.to_vec();
        let c = centroids.len() / d;
        self.graph.clear();
        self.graph.push(vec![Vec::new(); c]);
        self.levels.clear();
        self.max_layer = 0;
        self.entry_point = 0;
        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        for i in 0..c {
            let lvl = if i == 0 { 0 } else { self.assign_level(&mut rng) };
            self.insert(i as u32, lvl);
        }
        Ok(())
    }

    fn top_lists(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        assert_eq!(query.len(), self.d);
        let c = self.centroids.len() / self.d;
        let nprobe = nprobe.min(c);
        if c == 0 {
            return Vec::new();
        }
        // Greedy descent from top layer to layer 1.
        let mut ep = self.entry_point;
        for l in (1..=self.max_layer).rev() {
            ep = self.greedy(query, ep, l);
        }
        let ef = self.cfg.ef_search.max(nprobe);
        let mut res = self.search_layer(query, ep, ef, 0);
        res.truncate(nprobe);
        res.into_iter().map(|(_, i)| i).collect()
    }

    fn name(&self) -> &'static str {
        "HnswCoarse"
    }

    fn extra_bytes(&self) -> usize {
        // 4 bytes per edge (u32) summed across layers.
        let mut bytes = 0usize;
        for layer in &self.graph {
            for node in layer {
                bytes += node.len() * 4;
            }
        }
        bytes + self.levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatCoarse;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn gen(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn hnsw_matches_flat_on_small() {
        let d = 16;
        let c = 128;
        let centroids = gen(c, d, 1);
        let mut flat = FlatCoarse::new();
        flat.fit(&centroids, d).unwrap();
        let mut hnsw = HnswCoarse::with_config(HnswConfig {
            ef_construction: 100,
            ef_search: 64,
            ..Default::default()
        });
        hnsw.fit(&centroids, d).unwrap();
        let nq = 40;
        let queries = gen(nq, d, 2);
        let nprobe = 8;
        let mut agree = 0usize;
        let mut total = 0usize;
        for q in queries.chunks_exact(d) {
            let a: std::collections::HashSet<u32> =
                flat.top_lists(q, nprobe).into_iter().collect();
            let b: Vec<u32> = hnsw.top_lists(q, nprobe);
            for id in b {
                if a.contains(&id) {
                    agree += 1;
                }
            }
            total += nprobe;
        }
        let recall = agree as f64 / total as f64;
        assert!(
            recall > 0.90,
            "HnswCoarse recall {:.3} vs FlatCoarse too low",
            recall
        );
    }

    #[test]
    fn hnsw_top1_matches_flat() {
        let d = 8;
        let c = 64;
        let centroids = gen(c, d, 3);
        let mut flat = FlatCoarse::new();
        flat.fit(&centroids, d).unwrap();
        let mut hnsw = HnswCoarse::with_config(HnswConfig {
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        });
        hnsw.fit(&centroids, d).unwrap();
        let queries = gen(20, d, 4);
        let mut hits = 0;
        for q in queries.chunks_exact(d) {
            if flat.top_lists(q, 1)[0] == hnsw.top_lists(q, 1)[0] {
                hits += 1;
            }
        }
        assert!(hits >= 18, "top-1 recall {}/20 too low", hits);
    }
}
