//! ruvector-deg: Dynamic Exploration Graph (DEG) ANN index.
//!
//! Single-layer proximity graph with greedy beam search and a
//! Relative-Neighborhood-Graph (RNG) edge-optimization rule. Inspired by
//! Hülsmeier et al., "Dynamic Exploration Graph" (2024), and the
//! incremental edge-optimization line of work in the NSW / HNSW family.
//!
//! Three backends share an [`AnnIndex`] trait so they can be benchmarked
//! and swapped:
//!   * [`BruteForce`]      — exact baseline.
//!   * [`KnnGraph`]        — static k-NN graph with greedy beam search.
//!   * [`Deg`]             — dynamic graph with RNG pruning on insert.

use std::collections::{BinaryHeap, HashSet};

pub type Vector = Vec<f32>;

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

pub trait AnnIndex {
    fn insert(&mut self, v: Vector);
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Resident memory estimate in bytes (vectors + graph adjacency).
    fn mem_bytes(&self) -> usize;
}

// ---------------- BruteForce ----------------

#[derive(Default)]
pub struct BruteForce {
    pub data: Vec<Vector>,
    dim: usize,
}

impl BruteForce {
    pub fn new(dim: usize) -> Self {
        Self { data: Vec::new(), dim }
    }
}

impl AnnIndex for BruteForce {
    fn insert(&mut self, v: Vector) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.push(v);
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut heap: BinaryHeap<MaxHeapEntry> = BinaryHeap::with_capacity(k + 1);
        for (i, v) in self.data.iter().enumerate() {
            let d = l2_sq(q, v);
            if heap.len() < k {
                heap.push(MaxHeapEntry { id: i, dist: d });
            } else if d < heap.peek().unwrap().dist {
                heap.pop();
                heap.push(MaxHeapEntry { id: i, dist: d });
            }
        }
        let mut out: Vec<(usize, f32)> = heap.into_iter().map(|e| (e.id, e.dist)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }
    fn len(&self) -> usize { self.data.len() }
    fn mem_bytes(&self) -> usize {
        self.data.len() * self.dim * std::mem::size_of::<f32>()
    }
}

// ---------------- Max-heap helper ----------------
// Used for keeping the *worst* of the top-k so we can pop it cheaply.

#[derive(Clone, Copy)]
struct MaxHeapEntry { id: usize, dist: f32 }
impl Eq for MaxHeapEntry {}
impl PartialEq for MaxHeapEntry { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Ord for MaxHeapEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for MaxHeapEntry { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }

// Min-heap entry for the beam frontier.
#[derive(Clone, Copy)]
struct MinHeapEntry { id: usize, dist: f32 }
impl Eq for MinHeapEntry {}
impl PartialEq for MinHeapEntry { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Ord for MinHeapEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.dist.partial_cmp(&self.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for MinHeapEntry { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }

// ---------------- Shared graph search ----------------

fn beam_search(
    data: &[Vector],
    adj: &[Vec<u32>],
    entry: usize,
    q: &[f32],
    ef: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    beam_search_multi(data, adj, &[entry], q, ef, k)
}

fn beam_search_multi(
    data: &[Vector],
    adj: &[Vec<u32>],
    entries: &[usize],
    q: &[f32],
    ef: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    if data.is_empty() { return Vec::new(); }
    let mut visited = HashSet::with_capacity(ef * 4);
    let mut frontier: BinaryHeap<MinHeapEntry> = BinaryHeap::new();
    let mut topk: BinaryHeap<MaxHeapEntry> = BinaryHeap::with_capacity(ef + 1);

    for &entry in entries {
        if entry >= data.len() { continue; }
        if !visited.insert(entry) { continue; }
        let d0 = l2_sq(q, &data[entry]);
        frontier.push(MinHeapEntry { id: entry, dist: d0 });
        topk.push(MaxHeapEntry { id: entry, dist: d0 });
        if topk.len() > ef { topk.pop(); }
    }

    while let Some(cur) = frontier.pop() {
        let worst = topk.peek().map(|e| e.dist).unwrap_or(f32::INFINITY);
        if cur.dist > worst && topk.len() >= ef { break; }
        for &n in &adj[cur.id] {
            let n = n as usize;
            if !visited.insert(n) { continue; }
            let d = l2_sq(q, &data[n]);
            if topk.len() < ef || d < topk.peek().unwrap().dist {
                topk.push(MaxHeapEntry { id: n, dist: d });
                if topk.len() > ef { topk.pop(); }
                frontier.push(MinHeapEntry { id: n, dist: d });
            }
        }
    }

    let mut out: Vec<(usize, f32)> = topk.into_iter().map(|e| (e.id, e.dist)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.truncate(k);
    out
}

// ---------------- KnnGraph (static k-NN baseline) ----------------

pub struct KnnGraph {
    data: Vec<Vector>,
    adj: Vec<Vec<u32>>,
    dim: usize,
    pub ef_search: usize,
    pub k_build: usize,
}

impl KnnGraph {
    pub fn new(dim: usize, k_build: usize, ef_search: usize) -> Self {
        Self { data: Vec::new(), adj: Vec::new(), dim, ef_search, k_build }
    }

    /// Rebuild adjacency from scratch using exact k-NN (O(N^2)). Fine for
    /// the benchmark sizes we run; documented as the baseline.
    pub fn build(&mut self) {
        let n = self.data.len();
        let k = self.k_build;
        self.adj = vec![Vec::with_capacity(k); n];
        for i in 0..n {
            let mut heap: BinaryHeap<MaxHeapEntry> = BinaryHeap::with_capacity(k + 1);
            for j in 0..n {
                if i == j { continue; }
                let d = l2_sq(&self.data[i], &self.data[j]);
                if heap.len() < k {
                    heap.push(MaxHeapEntry { id: j, dist: d });
                } else if d < heap.peek().unwrap().dist {
                    heap.pop();
                    heap.push(MaxHeapEntry { id: j, dist: d });
                }
            }
            self.adj[i] = heap.into_iter().map(|e| e.id as u32).collect();
        }
    }
}

impl AnnIndex for KnnGraph {
    fn insert(&mut self, v: Vector) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.push(v);
        self.adj.push(Vec::new());
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() { return Vec::new(); }
        let n = self.data.len();
        let m = 8usize.min(n);
        let stride = (n / m.max(1)).max(1);
        let entries: Vec<usize> = (0..m).map(|i| (i * stride) % n).collect();
        beam_search_multi(&self.data, &self.adj, &entries, q, self.ef_search.max(k), k)
    }
    fn len(&self) -> usize { self.data.len() }
    fn mem_bytes(&self) -> usize {
        let vecs = self.data.len() * self.dim * std::mem::size_of::<f32>();
        let edges: usize = self.adj.iter().map(|a| a.len() * std::mem::size_of::<u32>()).sum();
        vecs + edges
    }
}

// ---------------- DEG (Dynamic Exploration Graph) ----------------

pub struct Deg {
    pub data: Vec<Vector>,
    pub adj: Vec<Vec<u32>>,
    dim: usize,
    pub max_degree: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    /// If true, prune incoming edges using the RNG (Relative
    /// Neighborhood Graph) rule for diversity.
    pub rng_pruning: bool,
    /// Number of distinct entry points sampled per query; multi-entry
    /// search is the standard DEG remedy for multi-modal data.
    pub n_entries: usize,
}

impl Deg {
    pub fn new(dim: usize, max_degree: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            data: Vec::new(),
            adj: Vec::new(),
            dim,
            max_degree,
            ef_construction,
            ef_search,
            rng_pruning: true,
            n_entries: 8,
        }
    }

    fn entries(&self) -> Vec<usize> {
        if self.data.is_empty() { return vec![]; }
        let n = self.data.len();
        let m = self.n_entries.max(1).min(n);
        // Deterministic stride sampling — cheap and reproducible.
        let stride = (n / m).max(1);
        (0..m).map(|i| (i * stride) % n).collect()
    }

    /// Apply Relative-Neighborhood-Graph pruning to a candidate set.
    /// Keep `c` only if no already-kept `r` satisfies d(new,r) < d(new,c)
    /// AND d(r,c) < d(new,c). Greedy, distance-order pass.
    fn rng_prune(&self, new_vec: &[f32], cands: Vec<(usize, f32)>, m: usize) -> Vec<(usize, f32)> {
        let mut kept: Vec<(usize, f32)> = Vec::with_capacity(m);
        for (cid, cdist) in cands {
            let mut dominated = false;
            for &(rid, _rdist) in &kept {
                let drc = l2_sq(&self.data[rid], &self.data[cid]);
                if drc < cdist {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                kept.push((cid, cdist));
                if kept.len() >= m { break; }
            }
            let _ = new_vec; // silence unused on no-debug builds
        }
        kept
    }

    fn add_undirected(&mut self, a: usize, b: usize) {
        if a == b { return; }
        if !self.adj[a].contains(&(b as u32)) { self.adj[a].push(b as u32); }
        if !self.adj[b].contains(&(a as u32)) { self.adj[b].push(a as u32); }
    }

    /// If a node exceeds `max_degree`, drop its farthest neighbor.
    /// On a tie we keep the older edge (stability).
    fn trim_neighbor(&mut self, node: usize) {
        if self.adj[node].len() <= self.max_degree { return; }
        let mut dists: Vec<(usize, f32)> = self.adj[node]
            .iter()
            .map(|&n| (n as usize, l2_sq(&self.data[node], &self.data[n as usize])))
            .collect();
        dists.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        // If RNG pruning is on, re-derive the kept set under RNG; otherwise
        // keep the M nearest. Either way we end up with <= M neighbors.
        let owned_vec = self.data[node].clone();
        let kept = if self.rng_pruning {
            self.rng_prune(&owned_vec, dists.clone(), self.max_degree)
        } else {
            dists.into_iter().take(self.max_degree).collect()
        };

        // Drop reverse edges from any neighbor that lost its slot.
        let new_set: HashSet<u32> = kept.iter().map(|(id, _)| *id as u32).collect();
        let old_set: HashSet<u32> = self.adj[node].iter().copied().collect();
        for dropped in old_set.difference(&new_set).copied() {
            self.adj[dropped as usize].retain(|&x| x != node as u32);
        }
        self.adj[node] = kept.into_iter().map(|(id, _)| id as u32).collect();
    }
}

impl AnnIndex for Deg {
    fn insert(&mut self, v: Vector) {
        debug_assert_eq!(v.len(), self.dim);
        let new_id = self.data.len();
        self.data.push(v.clone());
        self.adj.push(Vec::with_capacity(self.max_degree));
        if new_id == 0 { return; }

        let entries = self.entries();
        let candidates = beam_search_multi(
            &self.data,
            &self.adj,
            &entries,
            &v,
            self.ef_construction.max(self.max_degree),
            self.ef_construction,
        );

        let chosen = if self.rng_pruning {
            self.rng_prune(&v, candidates, self.max_degree)
        } else {
            candidates.into_iter().take(self.max_degree).collect()
        };

        for (nid, _d) in chosen {
            self.add_undirected(new_id, nid);
            self.trim_neighbor(nid);
        }
        self.trim_neighbor(new_id);
    }

    fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() { return Vec::new(); }
        let entries = self.entries();
        beam_search_multi(&self.data, &self.adj, &entries, q, self.ef_search.max(k), k)
    }

    fn len(&self) -> usize { self.data.len() }
    fn mem_bytes(&self) -> usize {
        let vecs = self.data.len() * self.dim * std::mem::size_of::<f32>();
        let edges: usize = self.adj.iter().map(|a| a.len() * std::mem::size_of::<u32>()).sum();
        vecs + edges
    }
}

// ---------------- Recall helper ----------------

/// recall@k of `pred` against ground-truth `truth` ids.
pub fn recall_at_k(pred: &[(usize, f32)], truth: &[(usize, f32)], k: usize) -> f32 {
    let kk = k.min(pred.len()).min(truth.len());
    if kk == 0 { return 0.0; }
    let gt: HashSet<usize> = truth.iter().take(kk).map(|(i, _)| *i).collect();
    let hit = pred.iter().take(kk).filter(|(i, _)| gt.contains(i)).count();
    hit as f32 / kk as f32
}
