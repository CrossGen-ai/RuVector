//! Minimal correct HNSW implementation with an explicit per-query
//! distance-computation counter. Squared-L2 distance is used (cheap and
//! monotone with Euclidean).
//!
//! Kept deliberately compact so the adaptive-ef logic is the focus of this
//! crate, not graph-construction wizardry. Throughput is competitive with
//! "from-scratch" reference implementations but not a fight against `hnswlib`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ml: f32, // level-multiplier; default 1/ln(M)
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ml: 1.0 / (m as f32).ln(),
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SearchStats {
    /// Number of distance computations performed during the query.
    pub dists: u64,
    /// Maximum heap size reached (proxy for memory pressure).
    pub max_heap: usize,
    /// Number of unique nodes visited.
    pub visited: usize,
}

#[derive(Clone, Copy, PartialEq)]
struct DistNode {
    dist: f32,
    node: u32,
}
impl Eq for DistNode {}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap by dist (BinaryHeap is max-heap)
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// reverse order wrapper for min-heap behaviour
#[derive(Clone, Copy, PartialEq)]
struct MinNode(DistNode);
impl Eq for MinNode {}
impl Ord for MinNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Hnsw {
    dim: usize,
    pub params: HnswParams,
    data: Vec<f32>,                // flat [n * dim]
    levels: Vec<u8>,               // top level for each node
    // adjacency: graph[level][node] -> Vec<u32> neighbours
    graph: Vec<Vec<Vec<u32>>>,
    entry_point: Option<u32>,
    rng: StdRng,
}

impl Hnsw {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self {
            dim,
            params,
            data: Vec::new(),
            levels: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            rng,
        }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }

    fn assign_level(&mut self) -> u8 {
        let r: f32 = self.rng.gen_range(1e-9_f32..1.0_f32);
        (-r.ln() * self.params.ml).floor().min(15.0) as u8
    }

    fn ensure_graph_depth(&mut self, level: u8) {
        while self.graph.len() <= level as usize {
            self.graph.push(Vec::new());
        }
        let n = self.levels.len();
        for l in 0..=level as usize {
            while self.graph[l].len() < n {
                self.graph[l].push(Vec::new());
            }
        }
    }

    pub fn insert(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.levels.len() as u32;
        let level = self.assign_level();
        self.data.extend_from_slice(v);
        self.levels.push(level);
        self.ensure_graph_depth(level);

        let mut stats = SearchStats::default();
        let ep = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                return id;
            }
            Some(ep) => ep,
        };

        // descend from top to level+1 with ef=1
        let top = *self.levels.iter().max().unwrap() as usize;
        let mut cur = ep;
        let mut cur_d = self.dist(self.vector(cur), v, &mut stats);
        for l in (level as usize + 1..=top).rev() {
            let (n, nd) = self.greedy_step(cur, cur_d, v, l, &mut stats);
            cur = n;
            cur_d = nd;
        }

        // ef_construction search at each level we are present in, then connect
        let m = self.params.m;
        for l in (0..=level as usize).rev() {
            let m_max = if l == 0 { self.params.m_max0 } else { m };
            let mut candidates = self.search_layer(
                v,
                cur,
                cur_d,
                self.params.ef_construction,
                l,
                &mut stats,
            );
            // pick M closest as neighbours
            candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            let pick: Vec<u32> = candidates.iter().take(m).map(|c| c.node).collect();
            for &nb in &pick {
                self.graph[l][id as usize].push(nb);
                self.graph[l][nb as usize].push(id);
                // prune nb's neighbours if oversized
                if self.graph[l][nb as usize].len() > m_max {
                    self.prune(nb, l, m_max, &mut stats);
                }
            }
            if let Some(best) = candidates.first() {
                cur = best.node;
                cur_d = best.dist;
            }
        }

        if level > self.levels[self.entry_point.unwrap() as usize] {
            self.entry_point = Some(id);
        }
        id
    }

    fn prune(&mut self, node: u32, level: usize, m_max: usize, stats: &mut SearchStats) {
        let v = self.vector(node).to_vec();
        let neighbours = self.graph[level][node as usize].clone();
        let mut scored: Vec<DistNode> = neighbours
            .iter()
            .map(|&n| DistNode { node: n, dist: self.dist(self.vector(n), &v, stats) })
            .collect();
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        scored.truncate(m_max);
        self.graph[level][node as usize] = scored.into_iter().map(|s| s.node).collect();
    }

    fn greedy_step(
        &self,
        start: u32,
        start_d: f32,
        q: &[f32],
        level: usize,
        stats: &mut SearchStats,
    ) -> (u32, f32) {
        let mut cur = start;
        let mut cur_d = start_d;
        loop {
            let mut changed = false;
            // bounds: level may not exist or node may not be in this layer yet
            if level >= self.graph.len() || (cur as usize) >= self.graph[level].len() {
                break;
            }
            let nbrs = self.graph[level][cur as usize].clone();
            for nb in nbrs {
                let d = self.dist(self.vector(nb), q, stats);
                if d < cur_d {
                    cur_d = d;
                    cur = nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        (cur, cur_d)
    }

    fn search_layer(
        &self,
        q: &[f32],
        ep: u32,
        ep_d: f32,
        ef: usize,
        level: usize,
        stats: &mut SearchStats,
    ) -> Vec<DistNode> {
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(ep);
        let mut candidates: BinaryHeap<MinNode> = BinaryHeap::new();
        let mut top: BinaryHeap<DistNode> = BinaryHeap::new();
        candidates.push(MinNode(DistNode { node: ep, dist: ep_d }));
        top.push(DistNode { node: ep, dist: ep_d });

        while let Some(MinNode(c)) = candidates.pop() {
            if let Some(worst) = top.peek() {
                if c.dist > worst.dist && top.len() >= ef {
                    break;
                }
            }
            if level >= self.graph.len() || (c.node as usize) >= self.graph[level].len() {
                continue;
            }
            for &nb in &self.graph[level][c.node as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = self.dist(self.vector(nb), q, stats);
                let push = top.len() < ef
                    || top.peek().map(|t| d < t.dist).unwrap_or(false);
                if push {
                    candidates.push(MinNode(DistNode { node: nb, dist: d }));
                    top.push(DistNode { node: nb, dist: d });
                    if top.len() > ef {
                        top.pop();
                    }
                }
            }
            if candidates.len() > stats.max_heap {
                stats.max_heap = candidates.len();
            }
        }
        stats.visited = visited.len();
        top.into_sorted_vec()
    }

    /// Standard HNSW knn search with explicit `ef`. Returns ids sorted by
    /// distance (ascending) along with the search statistics.
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> (Vec<(u32, f32)>, SearchStats) {
        assert_eq!(q.len(), self.dim);
        let mut stats = SearchStats::default();
        let ep = match self.entry_point {
            None => return (Vec::new(), stats),
            Some(ep) => ep,
        };
        let mut cur = ep;
        let mut cur_d = self.dist(self.vector(cur), q, &mut stats);
        let top_level = *self.levels.iter().max().unwrap() as usize;
        for l in (1..=top_level).rev() {
            let (n, nd) = self.greedy_step(cur, cur_d, q, l, &mut stats);
            cur = n;
            cur_d = nd;
        }
        let ef_eff = ef.max(k);
        let cands = self.search_layer(q, cur, cur_d, ef_eff, 0, &mut stats);
        let mut out: Vec<(u32, f32)> = cands.into_iter().map(|c| (c.node, c.dist)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        (out, stats)
    }

    /// Exposed for the predictor: descend down to layer 1, return the level-0
    /// entry point and its distance to `q`. Counts distances into `stats`.
    pub fn descend_to_layer0(&self, q: &[f32], stats: &mut SearchStats) -> Option<(u32, f32)> {
        let ep = self.entry_point?;
        let mut cur = ep;
        let mut cur_d = self.dist(self.vector(cur), q, stats);
        let top_level = *self.levels.iter().max().unwrap() as usize;
        for l in (1..=top_level).rev() {
            let (n, nd) = self.greedy_step(cur, cur_d, q, l, stats);
            cur = n;
            cur_d = nd;
        }
        Some((cur, cur_d))
    }

    /// Number of layer-0 neighbours of `node` (useful as a hardness feature).
    pub fn out_degree0(&self, node: u32) -> usize {
        if self.graph.is_empty() {
            return 0;
        }
        self.graph[0]
            .get(node as usize)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    fn dist(&self, a: &[f32], b: &[f32], stats: &mut SearchStats) -> f32 {
        stats.dists += 1;
        sq_l2(a, b)
    }
}

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Exact brute-force knn — used to build ground truth.
pub fn brute_knn(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = data.len() / dim;
    let mut all: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| {
            let s = i as usize * dim;
            (i, sq_l2(&data[s..s + dim], q))
        })
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(k);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect()
    }

    #[test]
    fn hnsw_finds_nearest() {
        let dim = 16;
        let n = 800;
        let data = random_vecs(n, dim, 42);
        let mut h = Hnsw::new(dim, HnswParams::default());
        for i in 0..n {
            h.insert(&data[i * dim..(i + 1) * dim]);
        }
        let q = &data[7 * dim..8 * dim];
        let (res, stats) = h.search(q, 5, 64);
        assert_eq!(res[0].0, 7);
        assert_eq!(res[0].1, 0.0);
        assert!(stats.dists > 0);
    }

    #[test]
    fn ef_increases_recall() {
        let dim = 8;
        let n = 1000;
        let data = random_vecs(n, dim, 7);
        let mut h = Hnsw::new(dim, HnswParams::default());
        for i in 0..n {
            h.insert(&data[i * dim..(i + 1) * dim]);
        }
        let queries = random_vecs(50, dim, 99);
        let k = 10;
        let mut r_low = 0usize;
        let mut r_high = 0usize;
        for qi in 0..50 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt: std::collections::HashSet<u32> =
                brute_knn(&data, dim, q, k).into_iter().map(|x| x.0).collect();
            let (lo, _) = h.search(q, k, 16);
            let (hi, _) = h.search(q, k, 128);
            r_low += lo.iter().filter(|(i, _)| gt.contains(i)).count();
            r_high += hi.iter().filter(|(i, _)| gt.contains(i)).count();
        }
        assert!(
            r_high >= r_low,
            "recall@128 ({}) should be >= recall@16 ({})",
            r_high,
            r_low
        );
    }
}
