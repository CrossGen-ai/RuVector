//! Query-Adaptive Frequency-Reordered HNSW Neighbor Lists.
//!
//! Core idea: in a small-world graph, greedy best-first search touches a
//! small tail of high-hub edges disproportionately often. Storing each
//! node's neighbor list in *hit-frequency* order (learned from a short
//! warmup of representative queries), rather than the usual insertion
//! or distance order, aligns the hottest edges with the earliest cache
//! lines of the neighbor array. That yields better prefetch coverage
//! and, when combined with distance-based early abandon inside the
//! neighbor scan, fewer distance computations per query at fixed `ef`.
//!
//! This crate ships:
//! - `Index`: a minimal single-layer NSW graph (enough to isolate the
//!   ordering effect; a full HNSW is the same effect times L layers).
//! - Three search variants sharing one code path via `NeighborOrder`:
//!   `Natural` (insertion order), `FreqReorder` (warmup-driven), and
//!   `FreqReorderPrefetch` (freq order + software prefetch of the first
//!   `PREFETCH_K` neighbors).
//! - A distance-early-abandon inner loop so ordering actually matters.

use std::collections::BinaryHeap;
use std::cmp::{Ordering, Reverse};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborOrder {
    Natural,
    FreqReorder,
    FreqReorderPrefetch,
}

/// Search-time knobs.
#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub ef: usize,
    pub k: usize,
    pub order: NeighborOrder,
}

/// Build-time knobs.
#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    pub m: usize,      // neighbors per node
    pub ef_build: usize,
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { m: 16, ef_build: 64, seed: 0xC0FFEE }
    }
}

/// Prefetch this many neighbors ahead when using FreqReorderPrefetch.
const PREFETCH_K: usize = 4;

/// A flat L2 vector store.
#[derive(Clone)]
pub struct VectorStore {
    dim: usize,
    data: Vec<f32>,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self { Self { dim, data: Vec::new() } }
    pub fn len(&self) -> usize { self.data.len() / self.dim }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
    }
    #[inline]
    pub fn get(&self, id: NodeId) -> &[f32] {
        let i = id as usize * self.dim;
        &self.data[i..i + self.dim]
    }
    /// Squared L2, no allocations, with early abandon at `bound`.
    /// Returns f32::INFINITY once the running sum would exceed `bound`.
    #[inline]
    pub fn l2_early(&self, a: NodeId, q: &[f32], bound: f32) -> f32 {
        let av = self.get(a);
        debug_assert_eq!(av.len(), q.len());
        let mut sum = 0.0f32;
        // Unrolled by 4 keeps the compiler generating tight SIMD-ish code
        // without pulling in an intrinsics dep.
        let n = av.len();
        let mut i = 0;
        while i + 4 <= n {
            let d0 = av[i] - q[i];
            let d1 = av[i + 1] - q[i + 1];
            let d2 = av[i + 2] - q[i + 2];
            let d3 = av[i + 3] - q[i + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            if sum > bound { return f32::INFINITY; }
            i += 4;
        }
        while i < n {
            let d = av[i] - q[i];
            sum += d * d;
            i += 1;
        }
        sum
    }
    #[inline]
    pub fn l2(&self, a: NodeId, q: &[f32]) -> f32 {
        self.l2_early(a, q, f32::INFINITY)
    }
}

/// A minimal single-layer NSW graph.
pub struct Index {
    pub store: VectorStore,
    pub adj: Vec<Vec<NodeId>>,   // per-node neighbor list
    pub freq: Vec<Vec<u32>>,     // per-edge hit counter (parallel to adj)
    entry: NodeId,
    params: BuildParams,
    /// Total distance computations across the last search batch,
    /// exposed for benchmarks.
    pub dist_count: std::cell::Cell<u64>,
}

impl Index {
    pub fn new(dim: usize, params: BuildParams) -> Self {
        Self {
            store: VectorStore::new(dim),
            adj: Vec::new(),
            freq: Vec::new(),
            entry: 0,
            params,
            dist_count: std::cell::Cell::new(0),
        }
    }

    pub fn len(&self) -> usize { self.store.len() }
    pub fn is_empty(&self) -> bool { self.store.is_empty() }

    /// Build the graph by streaming inserts.
    pub fn build(&mut self, vectors: &[Vec<f32>]) {
        for v in vectors { self.insert(v); }
    }

    /// Insert one vector.
    pub fn insert(&mut self, v: &[f32]) -> NodeId {
        let id = self.store.len() as NodeId;
        self.store.push(v);
        self.adj.push(Vec::with_capacity(self.params.m));
        self.freq.push(Vec::with_capacity(self.params.m));
        if id == 0 {
            self.entry = 0;
            return id;
        }
        // Search for M nearest existing nodes using the current graph.
        let cands = self.search_layer(v, self.entry, self.params.ef_build);
        let mut chosen: Vec<(f32, NodeId)> = cands.into_iter().collect();
        chosen.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        chosen.truncate(self.params.m);
        for (_, nb) in &chosen {
            // Directed both ways; trim reverse side if it overflows M via
            // simple distance-based pruning.
            self.adj[id as usize].push(*nb);
            self.freq[id as usize].push(0);
            let nb_idx = *nb as usize;
            if self.adj[nb_idx].len() < self.params.m {
                self.adj[nb_idx].push(id);
                self.freq[nb_idx].push(0);
            } else {
                // Replace farthest existing neighbor if `id` is closer.
                let nb_vec = self.store.get(*nb).to_vec();
                let d_new = self.store.l2(id, &nb_vec);
                let (mut worst_i, mut worst_d) = (0usize, f32::NEG_INFINITY);
                for (k, &m) in self.adj[nb_idx].iter().enumerate() {
                    let dm = self.store.l2(m, &nb_vec);
                    if dm > worst_d { worst_d = dm; worst_i = k; }
                }
                if d_new < worst_d {
                    self.adj[nb_idx][worst_i] = id;
                    self.freq[nb_idx][worst_i] = 0;
                }
            }
        }
        // Long-range random links keep a single-layer NSW navigable
        // across clusters (a rougher stand-in for HNSW's upper layers).
        let long_range = (self.params.m / 4).max(2);
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.params.seed ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        for _ in 0..long_range {
            if id == 0 { break; }
            let peer = rng.gen_range(0..id) as NodeId;
            if !self.adj[id as usize].contains(&peer) {
                self.adj[id as usize].push(peer);
                self.freq[id as usize].push(0);
                if !self.adj[peer as usize].contains(&id) {
                    self.adj[peer as usize].push(id);
                    self.freq[peer as usize].push(0);
                }
            }
        }
        id
    }

    /// Greedy layer search used during build. Returns (dist, id) pairs.
    fn search_layer(&self, q: &[f32], entry: NodeId, ef: usize) -> Vec<(f32, NodeId)> {
        let mut visited = vec![false; self.store.len()];
        // Frontier: min-heap on distance (Reverse).
        let mut frontier: BinaryHeap<Reverse<Ordered>> = BinaryHeap::new();
        // Results: max-heap on distance so we can pop the worst.
        let mut results: BinaryHeap<Ordered> = BinaryHeap::new();

        let d0 = self.store.l2(entry, q);
        frontier.push(Reverse(Ordered(d0, entry)));
        results.push(Ordered(d0, entry));
        visited[entry as usize] = true;

        while let Some(Reverse(Ordered(d, cur))) = frontier.pop() {
            if let Some(top) = results.peek() {
                if d > top.0 && results.len() >= ef { break; }
            }
            for &nb in &self.adj[cur as usize] {
                let nbi = nb as usize;
                if visited[nbi] { continue; }
                visited[nbi] = true;
                let dn = self.store.l2(nb, q);
                if results.len() < ef || dn < results.peek().map(|t| t.0).unwrap_or(f32::INFINITY) {
                    frontier.push(Reverse(Ordered(dn, nb)));
                    results.push(Ordered(dn, nb));
                    if results.len() > ef { results.pop(); }
                }
            }
        }
        results.into_sorted_vec().into_iter().map(|Ordered(d, id)| (d, id)).collect()
    }

    /// Public query. `order` selects the neighbor-scan variant.
    pub fn search(&self, q: &[f32], params: SearchParams) -> Vec<(f32, NodeId)> {
        let mut dc: u64 = 0;
        let out = self.search_impl(q, params, &mut dc, /*record_freq=*/ false);
        self.dist_count.set(self.dist_count.get() + dc);
        out
    }

    /// Warmup pass: run queries, recording per-edge traversal counts.
    /// Must be called BEFORE `reorder_by_frequency`.
    pub fn warmup(&mut self, queries: &[Vec<f32>], ef: usize) {
        let sp = SearchParams { ef, k: ef, order: NeighborOrder::Natural };
        // We need &mut self because we bump `self.freq`, but the
        // read-side of search_impl only needs `&self`. Cheat with a
        // manual inline copy that writes to `self.freq` on the hit path.
        for q in queries { self.warmup_query(q, sp); }
    }

    fn warmup_query(&mut self, q: &[f32], sp: SearchParams) {
        let n = self.store.len();
        let mut visited = vec![false; n];
        let mut frontier: BinaryHeap<Reverse<Ordered>> = BinaryHeap::new();
        let mut results: BinaryHeap<Ordered> = BinaryHeap::new();
        let d0 = self.store.l2(self.entry, q);
        frontier.push(Reverse(Ordered(d0, self.entry)));
        results.push(Ordered(d0, self.entry));
        visited[self.entry as usize] = true;

        while let Some(Reverse(Ordered(d, cur))) = frontier.pop() {
            if let Some(top) = results.peek() {
                if d > top.0 && results.len() >= sp.ef { break; }
            }
            let neigh_len = self.adj[cur as usize].len();
            for i in 0..neigh_len {
                let nb = self.adj[cur as usize][i];
                let nbi = nb as usize;
                if visited[nbi] { continue; }
                visited[nbi] = true;
                let bound = results.peek().map(|t| t.0).unwrap_or(f32::INFINITY);
                let dn = if results.len() >= sp.ef {
                    self.store.l2_early(nb, q, bound)
                } else {
                    self.store.l2(nb, q)
                };
                if dn.is_finite() {
                    // Edge (cur -> nb) contributed a real distance compute.
                    self.freq[cur as usize][i] = self.freq[cur as usize][i].saturating_add(1);
                    if results.len() < sp.ef || dn < bound {
                        frontier.push(Reverse(Ordered(dn, nb)));
                        results.push(Ordered(dn, nb));
                        if results.len() > sp.ef { results.pop(); }
                    }
                }
            }
        }
    }

    /// Sort each node's neighbor list by descending hit-frequency.
    /// Stable within ties so the natural order acts as the tie-breaker.
    pub fn reorder_by_frequency(&mut self) {
        for i in 0..self.adj.len() {
            let n = self.adj[i].len();
            let mut pairs: Vec<(u32, NodeId)> = (0..n)
                .map(|k| (self.freq[i][k], self.adj[i][k]))
                .collect();
            // Descending by freq, stable.
            pairs.sort_by(|a, b| b.0.cmp(&a.0));
            for k in 0..n {
                self.freq[i][k] = pairs[k].0;
                self.adj[i][k] = pairs[k].1;
            }
        }
    }

    /// Reset traversal counters (for measuring dist_count on a fresh run).
    pub fn reset_dist_count(&self) { self.dist_count.set(0); }

    fn search_impl(
        &self,
        q: &[f32],
        sp: SearchParams,
        dc: &mut u64,
        _record_freq: bool,
    ) -> Vec<(f32, NodeId)> {
        let n = self.store.len();
        let mut visited = vec![false; n];
        let mut frontier: BinaryHeap<Reverse<Ordered>> = BinaryHeap::new();
        let mut results: BinaryHeap<Ordered> = BinaryHeap::new();

        let d0 = self.store.l2(self.entry, q);
        *dc += 1;
        frontier.push(Reverse(Ordered(d0, self.entry)));
        results.push(Ordered(d0, self.entry));
        visited[self.entry as usize] = true;

        while let Some(Reverse(Ordered(d, cur))) = frontier.pop() {
            if let Some(top) = results.peek() {
                if d > top.0 && results.len() >= sp.ef { break; }
            }
            let nbrs = &self.adj[cur as usize];
            // Software prefetch for the FreqReorderPrefetch variant. We
            // touch the first PREFETCH_K neighbor vectors so the loop
            // that follows finds them in L1.
            if matches!(sp.order, NeighborOrder::FreqReorderPrefetch) {
                let take = nbrs.len().min(PREFETCH_K);
                for j in 0..take {
                    let nbi = nbrs[j] as usize;
                    // The vector data lives contiguously in store.data;
                    // touching the head of the slice is a portable
                    // stand-in for _mm_prefetch and works on any target.
                    let _ = unsafe {
                        std::ptr::read_volatile(
                            self.store.data.as_ptr().add(nbi * self.store.dim),
                        )
                    };
                }
            }
            for &nb in nbrs {
                let nbi = nb as usize;
                if visited[nbi] { continue; }
                visited[nbi] = true;
                let bound = if results.len() >= sp.ef {
                    results.peek().map(|t| t.0).unwrap_or(f32::INFINITY)
                } else {
                    f32::INFINITY
                };
                let dn = self.store.l2_early(nb, q, bound);
                *dc += 1;
                if dn.is_finite() {
                    if results.len() < sp.ef || dn < bound {
                        frontier.push(Reverse(Ordered(dn, nb)));
                        results.push(Ordered(dn, nb));
                        if results.len() > sp.ef { results.pop(); }
                    }
                }
            }
        }

        let mut out: Vec<(f32, NodeId)> = results
            .into_sorted_vec()
            .into_iter()
            .map(|Ordered(d, id)| (d, id))
            .collect();
        out.truncate(sp.k);
        out
    }
}

/// Distance-then-id tuple ordered by distance (max-heap by default).
#[derive(Copy, Clone, Debug)]
struct Ordered(f32, NodeId);
impl PartialEq for Ordered {
    fn eq(&self, o: &Self) -> bool { self.0 == o.0 && self.1 == o.1 }
}
impl Eq for Ordered {}
impl Ord for Ordered {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal).then(self.1.cmp(&o.1))
    }
}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

/// Brute-force ground truth used by benchmarks.
pub fn brute_force_topk(store: &VectorStore, q: &[f32], k: usize) -> Vec<(f32, NodeId)> {
    let n = store.len();
    let mut all: Vec<(f32, NodeId)> = (0..n as NodeId)
        .map(|i| (store.l2(i, q), i))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    all.truncate(k);
    all
}

/// Recall@k versus a brute-force truth set.
pub fn recall(pred: &[(f32, NodeId)], truth: &[(f32, NodeId)]) -> f32 {
    if truth.is_empty() { return 1.0; }
    let mut set = std::collections::HashSet::new();
    for (_, id) in truth { set.insert(*id); }
    let mut hits = 0usize;
    for (_, id) in pred { if set.contains(id) { hits += 1; } }
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn gen(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
    }

    #[test]
    fn build_and_search() {
        let vs = gen(500, 16, 1);
        let mut idx = Index::new(16, BuildParams { m: 8, ef_build: 32, seed: 0 });
        idx.build(&vs);
        let q = &vs[3];
        let out = idx.search(q, SearchParams { ef: 16, k: 5, order: NeighborOrder::Natural });
        assert_eq!(out.len(), 5);
        // The vector itself should be the top hit.
        assert_eq!(out[0].1, 3);
    }

    #[test]
    fn freq_reorder_preserves_recall() {
        let vs = gen(800, 24, 7);
        let mut idx = Index::new(24, BuildParams { m: 12, ef_build: 48, seed: 0 });
        idx.build(&vs);
        let train: Vec<Vec<f32>> = vs[..200].to_vec();
        idx.warmup(&train, 24);
        idx.reorder_by_frequency();
        let q = &vs[42];
        let out_nat = idx.search(q, SearchParams { ef: 24, k: 10, order: NeighborOrder::Natural });
        let out_fr = idx.search(q, SearchParams { ef: 24, k: 10, order: NeighborOrder::FreqReorder });
        let truth = brute_force_topk(&idx.store, q, 10);
        // Both variants should recall the same set; reorder is only
        // supposed to accelerate the scan, not change greedy semantics
        // once ef is large enough.
        assert!(recall(&out_nat, &truth) >= 0.7);
        assert!(recall(&out_fr, &truth) >= 0.7);
    }
}
