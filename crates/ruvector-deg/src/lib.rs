//! Dynamic Exploration Graph (DEG)
//!
//! A continuously self-optimizing proximity graph index for high-recall
//! approximate nearest-neighbor search. Inspired by Hezel, Schall et al.'s
//! 2023-2024 line of work showing that periodic *edge optimization* on a
//! fixed-degree graph can produce search quality comparable or superior to
//! HNSW at a given degree budget, especially under streaming insertion.
//!
//! Three backends are exposed behind a single [`AnnIndex`] trait so the
//! research benchmarks can compare them on identical data:
//!
//! * [`RandomGraph`] — degree-D random graph (baseline).
//! * [`NswGraph`] — greedy navigable small-world build (HNSW layer-0-like).
//! * [`DegGraph`] — NSW build + RNG-style pruning + edge-swap optimization.
//!
//! All distances are squared Euclidean. Vectors are owned `Vec<f32>` and
//! stored contiguously in a flat arena.

use rand::Rng;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

pub mod metric;

/// Identifier for a vector in the index.
pub type NodeId = u32;

/// Common ANN index interface used by the three backends.
pub trait AnnIndex {
    /// Insert a vector. Returns its assigned [`NodeId`].
    fn insert(&mut self, vec: Vec<f32>) -> NodeId;
    /// Return the top-`k` approximate nearest neighbors of `query`.
    /// Result is sorted nearest-first as `(id, squared_distance)`.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)>;
    /// Number of vectors stored.
    fn len(&self) -> usize;
    /// Number of edges currently in the graph (sum of out-degrees).
    fn edge_count(&self) -> usize;
}

// -------------------------------------------------------------------------
// Heap entries
// -------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
struct DistNode {
    dist: f32,
    id: NodeId,
}

impl PartialEq for DistNode {
    fn eq(&self, other: &Self) -> bool { self.dist == other.dist && self.id == other.id }
}
impl Eq for DistNode {}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for DistNode {
    // Max-heap by distance; ties broken by id.
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// Min-heap wrapper (for candidate frontiers).
#[derive(Copy, Clone, Debug)]
struct MinDist(DistNode);
impl PartialEq for MinDist { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for MinDist {}
impl PartialOrd for MinDist { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MinDist { fn cmp(&self, o: &Self) -> Ordering { o.0.cmp(&self.0) } }

// -------------------------------------------------------------------------
// Shared storage
// -------------------------------------------------------------------------

/// Flat vector arena shared by all three backends.
pub struct Storage {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Storage {
    pub fn new(dim: usize) -> Self { Self { dim, data: Vec::new() } }
    pub fn push(&mut self, v: Vec<f32>) -> NodeId {
        assert_eq!(v.len(), self.dim, "vector dim mismatch");
        let id = (self.data.len() / self.dim) as NodeId;
        self.data.extend_from_slice(&v);
        id
    }
    pub fn get(&self, id: NodeId) -> &[f32] {
        let off = (id as usize) * self.dim;
        &self.data[off..off + self.dim]
    }
    pub fn len(&self) -> usize { self.data.len() / self.dim }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

// -------------------------------------------------------------------------
// Backend 1: Random degree-D graph
// -------------------------------------------------------------------------

pub struct RandomGraph {
    pub storage: Storage,
    pub degree: usize,
    pub edges: Vec<Vec<NodeId>>,
    rng_seed: u64,
}

impl RandomGraph {
    pub fn new(dim: usize, degree: usize, seed: u64) -> Self {
        Self { storage: Storage::new(dim), degree, edges: Vec::new(), rng_seed: seed }
    }
}

impl AnnIndex for RandomGraph {
    fn insert(&mut self, vec: Vec<f32>) -> NodeId {
        let id = self.storage.push(vec);
        let n = self.storage.len();
        let mut nbrs = Vec::with_capacity(self.degree);
        if n > 1 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed ^ (id as u64));
            use rand::SeedableRng;
            let take = self.degree.min(n - 1);
            while nbrs.len() < take {
                let c = rng.gen_range(0..(n as u32));
                if c != id && !nbrs.contains(&c) { nbrs.push(c); }
            }
            for &c in &nbrs {
                let cu = c as usize;
                if self.edges[cu].len() < self.degree { self.edges[cu].push(id); }
            }
        }
        self.edges.push(nbrs);
        id
    }
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        greedy_search(&self.storage, &self.edges, query, k, ef, 0)
    }
    fn len(&self) -> usize { self.storage.len() }
    fn edge_count(&self) -> usize { self.edges.iter().map(|e| e.len()).sum() }
}

// -------------------------------------------------------------------------
// Backend 2: NSW (greedy small-world) graph
// -------------------------------------------------------------------------

pub struct NswGraph {
    pub storage: Storage,
    pub degree: usize,
    pub ef_construction: usize,
    pub edges: Vec<Vec<NodeId>>,
}

impl NswGraph {
    pub fn new(dim: usize, degree: usize, ef_construction: usize) -> Self {
        Self { storage: Storage::new(dim), degree, ef_construction, edges: Vec::new() }
    }
}

impl AnnIndex for NswGraph {
    fn insert(&mut self, vec: Vec<f32>) -> NodeId {
        let id = self.storage.push(vec);
        if self.storage.len() == 1 { self.edges.push(Vec::new()); return id; }
        let q = self.storage.get(id).to_vec();
        let cands = greedy_search(&self.storage, &self.edges, &q, self.degree, self.ef_construction.max(self.degree), 0);
        let mut nbrs: Vec<NodeId> = cands.iter().map(|(i, _)| *i).collect();
        nbrs.truncate(self.degree);
        for &c in &nbrs {
            let cu = c as usize;
            self.edges[cu].push(id);
            if self.edges[cu].len() > self.degree {
                // Trim by distance: keep nearest `degree`.
                let cv = self.storage.get(c).to_vec();
                let mut scored: Vec<(NodeId, f32)> = self.edges[cu].iter()
                    .map(|&n| (n, metric::sq_l2(&cv, self.storage.get(n)))).collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                scored.truncate(self.degree);
                self.edges[cu] = scored.into_iter().map(|(i, _)| i).collect();
            }
        }
        self.edges.push(nbrs);
        id
    }
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        greedy_search(&self.storage, &self.edges, query, k, ef, 0)
    }
    fn len(&self) -> usize { self.storage.len() }
    fn edge_count(&self) -> usize { self.edges.iter().map(|e| e.len()).sum() }
}

// -------------------------------------------------------------------------
// Backend 3: Dynamic Exploration Graph
// -------------------------------------------------------------------------

/// Parameters controlling DEG construction and self-optimization.
#[derive(Clone, Copy, Debug)]
pub struct DegConfig {
    pub degree: usize,            // target out-degree D
    pub ef_construction: usize,   // candidate frontier size during insert
    pub extend_eps: f32,          // RNG pruning relaxation in (0, 1]
    pub optimize_every: usize,    // run edge-swap optimization every N inserts
    pub optimize_passes: usize,   // swap attempts per optimization run (× n)
    pub swap_eps: f32,            // accept swap if improvement > swap_eps
}

impl Default for DegConfig {
    fn default() -> Self {
        Self {
            degree: 16,
            ef_construction: 64,
            extend_eps: 0.30,
            optimize_every: 256,
            optimize_passes: 1,
            swap_eps: 0.0,
        }
    }
}

pub struct DegGraph {
    pub storage: Storage,
    pub cfg: DegConfig,
    pub edges: Vec<Vec<NodeId>>,
    inserts_since_opt: usize,
    rng_state: u64,
}

impl DegGraph {
    pub fn new(dim: usize, cfg: DegConfig) -> Self {
        Self {
            storage: Storage::new(dim),
            cfg,
            edges: Vec::new(),
            inserts_since_opt: 0,
            rng_state: 0xD06D06_C0FFEE,
        }
    }

    /// RNG-relaxed neighbor selection.
    ///
    /// Given a sorted list of candidates `(id, dist)` (nearest first), pick up
    /// to `degree` neighbors where each kept candidate `c` is closer to the
    /// query than to every already-kept neighbor scaled by `(1 - eps)`.
    /// `eps = 0` recovers strict RNG; `eps -> 1` recovers naive top-D.
    fn rng_prune(&self, _q: &[f32], cands: &[(NodeId, f32)]) -> Vec<NodeId> {
        let mut kept: Vec<NodeId> = Vec::with_capacity(self.cfg.degree);
        let one_minus = 1.0 - self.cfg.extend_eps;
        // Phase 1: diversity-aware (RNG-like) selection.
        for &(c, dq) in cands {
            if kept.len() >= self.cfg.degree { break; }
            let cv = self.storage.get(c);
            let mut ok = true;
            for &k in &kept {
                let kv = self.storage.get(k);
                let dck = metric::sq_l2(cv, kv);
                if dck * one_minus < dq { ok = false; break; }
            }
            if ok { kept.push(c); }
        }
        // Phase 2: if pruning was too aggressive (very common early in life),
        // backfill from the remaining nearest candidates. This preserves the
        // RNG diversity bias while guaranteeing connectivity.
        if kept.len() < self.cfg.degree {
            for &(c, _) in cands {
                if kept.len() >= self.cfg.degree { break; }
                if !kept.contains(&c) { kept.push(c); }
            }
        }
        kept
    }

    /// Edge-swap optimization: walk a random sample of nodes; for each node A
    /// look at neighbors-of-neighbors and try to replace its worst edge with
    /// a strictly closer candidate. This is the heart of "dynamic exploration":
    /// the graph keeps improving even after all data is inserted.
    fn optimize(&mut self, passes: usize) -> usize {
        let n = self.storage.len();
        if n < 3 { return 0; }
        let mut swaps = 0usize;
        for _ in 0..passes {
            for _ in 0..n {
                self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = ((self.rng_state >> 33) as u32) % (n as u32);
                let av = self.storage.get(a).to_vec();
                // Worst current edge of A
                let mut worst_idx = 0usize;
                let mut worst_d = f32::NEG_INFINITY;
                for (i, &nb) in self.edges[a as usize].iter().enumerate() {
                    let d = metric::sq_l2(&av, self.storage.get(nb));
                    if d > worst_d { worst_d = d; worst_idx = i; }
                }
                if !worst_d.is_finite() { continue; }
                // Candidate pool = 2-hop neighborhood
                let mut pool: Vec<NodeId> = Vec::with_capacity(64);
                for &nb in &self.edges[a as usize] {
                    for &nn in &self.edges[nb as usize] {
                        if nn != a && !self.edges[a as usize].contains(&nn) && !pool.contains(&nn) {
                            pool.push(nn);
                            if pool.len() >= 64 { break; }
                        }
                    }
                    if pool.len() >= 64 { break; }
                }
                // Try the closest candidate
                let mut best: Option<(NodeId, f32)> = None;
                for c in pool {
                    let d = metric::sq_l2(&av, self.storage.get(c));
                    if best.map_or(true, |(_, bd)| d < bd) {
                        best = Some((c, d));
                    }
                }
                if let Some((c, d)) = best {
                    if d + self.cfg.swap_eps < worst_d {
                        // Swap: replace worst with c; also patch reverse edge.
                        let old = self.edges[a as usize][worst_idx];
                        self.edges[a as usize][worst_idx] = c;
                        // Remove A from old's adjacency if present.
                        let ou = old as usize;
                        if let Some(p) = self.edges[ou].iter().position(|&x| x == a) {
                            self.edges[ou].swap_remove(p);
                        }
                        // Add A to c's adjacency, evicting c's worst if at capacity.
                        let cu = c as usize;
                        if self.edges[cu].len() < self.cfg.degree {
                            self.edges[cu].push(a);
                        } else {
                            let cv = self.storage.get(c).to_vec();
                            let (mut wi, mut wd) = (0usize, f32::NEG_INFINITY);
                            for (i, &nb) in self.edges[cu].iter().enumerate() {
                                let dd = metric::sq_l2(&cv, self.storage.get(nb));
                                if dd > wd { wd = dd; wi = i; }
                            }
                            let da = metric::sq_l2(&cv, &av);
                            if da < wd { self.edges[cu][wi] = a; }
                        }
                        swaps += 1;
                    }
                }
            }
        }
        swaps
    }

    /// Public hook so the bench can force a final optimization pass and
    /// report swap counts.
    pub fn optimize_all(&mut self, passes: usize) -> usize {
        self.optimize(passes)
    }
}

impl AnnIndex for DegGraph {
    fn insert(&mut self, vec: Vec<f32>) -> NodeId {
        let id = self.storage.push(vec);
        if self.storage.len() == 1 { self.edges.push(Vec::new()); return id; }
        let q = self.storage.get(id).to_vec();
        let cands = greedy_search(
            &self.storage,
            &self.edges,
            &q,
            self.cfg.degree * 2,
            self.cfg.ef_construction,
            0,
        );
        let nbrs = self.rng_prune(&q, &cands);
        for &c in &nbrs {
            let cu = c as usize;
            if self.edges[cu].len() < self.cfg.degree {
                self.edges[cu].push(id);
            } else {
                // Evict c's worst current edge if id is closer.
                let cv = self.storage.get(c).to_vec();
                let (mut wi, mut wd) = (0usize, f32::NEG_INFINITY);
                for (i, &nb) in self.edges[cu].iter().enumerate() {
                    let dd = metric::sq_l2(&cv, self.storage.get(nb));
                    if dd > wd { wd = dd; wi = i; }
                }
                let dn = metric::sq_l2(&cv, &q);
                if dn < wd { self.edges[cu][wi] = id; }
            }
        }
        self.edges.push(nbrs);
        self.inserts_since_opt += 1;
        if self.inserts_since_opt >= self.cfg.optimize_every {
            self.inserts_since_opt = 0;
            self.optimize(self.cfg.optimize_passes);
        }
        id
    }
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        greedy_search(&self.storage, &self.edges, query, k, ef, 0)
    }
    fn len(&self) -> usize { self.storage.len() }
    fn edge_count(&self) -> usize { self.edges.iter().map(|e| e.len()).sum() }
}

// -------------------------------------------------------------------------
// Shared greedy beam search
// -------------------------------------------------------------------------

fn greedy_search(
    storage: &Storage,
    edges: &[Vec<NodeId>],
    query: &[f32],
    k: usize,
    ef: usize,
    entry: NodeId,
) -> Vec<(NodeId, f32)> {
    use std::collections::HashSet;
    if storage.is_empty() { return Vec::new(); }
    let entry = entry.min((storage.len() - 1) as NodeId);
    let d0 = metric::sq_l2(query, storage.get(entry));
    let mut visited: HashSet<NodeId> = HashSet::new();
    visited.insert(entry);
    let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
    let mut results: BinaryHeap<DistNode> = BinaryHeap::new();
    candidates.push(MinDist(DistNode { dist: d0, id: entry }));
    results.push(DistNode { dist: d0, id: entry });
    let ef = ef.max(k);
    while let Some(MinDist(cur)) = candidates.pop() {
        let worst = results.peek().map(|n| n.dist).unwrap_or(f32::INFINITY);
        if cur.dist > worst && results.len() >= ef { break; }
        for &nb in &edges[cur.id as usize] {
            if !visited.insert(nb) { continue; }
            let d = metric::sq_l2(query, storage.get(nb));
            let worst = results.peek().map(|n| n.dist).unwrap_or(f32::INFINITY);
            if results.len() < ef || d < worst {
                candidates.push(MinDist(DistNode { dist: d, id: nb }));
                results.push(DistNode { dist: d, id: nb });
                if results.len() > ef { results.pop(); }
            }
        }
    }
    let mut out: Vec<(NodeId, f32)> = results.into_iter().map(|n| (n.id, n.dist)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out.truncate(k);
    out
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()).collect()
    }

    fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = data.iter().enumerate()
            .map(|(i, v)| (i as u32, metric::sq_l2(q, v))).collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn recall(got: &[(u32, f32)], gold: &[u32]) -> f32 {
        let g: std::collections::HashSet<u32> = gold.iter().copied().collect();
        got.iter().filter(|(i, _)| g.contains(i)).count() as f32 / gold.len() as f32
    }

    #[test]
    fn nsw_beats_random_in_recall() {
        let n = 600; let dim = 16; let k = 10;
        let data = synth(n, dim, 1);
        let queries = synth(40, dim, 2);
        let mut rg = RandomGraph::new(dim, 12, 7);
        let mut ns = NswGraph::new(dim, 12, 48);
        for v in &data { rg.insert(v.clone()); ns.insert(v.clone()); }
        let (mut rr, mut rn) = (0.0, 0.0);
        for q in &queries {
            let gold = brute_topk(&data, q, k);
            rr += recall(&rg.search(q, k, 48), &gold);
            rn += recall(&ns.search(q, k, 48), &gold);
        }
        rr /= queries.len() as f32; rn /= queries.len() as f32;
        assert!(rn > rr + 0.05, "NSW recall {rn} should exceed random {rr} by >0.05");
    }

    #[test]
    fn deg_optimize_improves_or_holds_recall() {
        let n = 800; let dim = 24; let k = 10;
        let data = synth(n, dim, 11);
        let queries = synth(50, dim, 12);
        let mut cfg = DegConfig::default();
        cfg.degree = 12; cfg.ef_construction = 64; cfg.optimize_every = usize::MAX;
        let mut deg = DegGraph::new(dim, cfg);
        for v in &data { deg.insert(v.clone()); }
        let mut before = 0.0;
        for q in &queries {
            let gold = brute_topk(&data, q, k);
            before += recall(&deg.search(q, k, 64), &gold);
        }
        before /= queries.len() as f32;
        let swaps = deg.optimize_all(4);
        let mut after = 0.0;
        for q in &queries {
            let gold = brute_topk(&data, q, k);
            after += recall(&deg.search(q, k, 64), &gold);
        }
        after /= queries.len() as f32;
        eprintln!("DEG recall before={before:.3} after={after:.3} swaps={swaps}");
        assert!(after + 0.02 >= before, "optimization should not collapse recall");
    }

    #[test]
    fn degree_bound_respected() {
        let n = 300; let dim = 8;
        let data = synth(n, dim, 3);
        let mut cfg = DegConfig::default();
        cfg.degree = 10; cfg.optimize_every = 64;
        let mut deg = DegGraph::new(dim, cfg);
        for v in &data { deg.insert(v.clone()); }
        for e in &deg.edges { assert!(e.len() <= 10); }
    }
}
