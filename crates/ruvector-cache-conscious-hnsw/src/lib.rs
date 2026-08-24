//! Cache-conscious HNSW-lite: flat single-layer navigable graph with
//! swappable node-ordering strategies (Insertion / BFS / Reverse-Cuthill-McKee).
//!
//! Design goals:
//!   * Nodes stored SoA: `vectors` is a flat `Vec<f32>` of length `n * dim`;
//!     `neighbors` is a flat `Vec<u32>` of length `n * max_degree`, so
//!     `permute()` is a cheap O(n * (dim + max_degree)) copy.
//!   * A single trait `NodeOrdering` returns a permutation `old_id -> new_id`;
//!     backends can evolve (BFS today, learned ordering tomorrow) without
//!     touching search.
//!   * Search is a plain greedy beam over `visited + heap`, identical for
//!     every ordering — the only variable is memory-access order.
//!
//! No unsafe. No SIMD intrinsics (portable). No mocks.

pub mod reorder;

use std::collections::BinaryHeap;

/// Distance = squared Euclidean.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[derive(Clone, Debug)]
pub struct FlatGraph {
    pub dim: usize,
    pub max_degree: usize,
    pub n: usize,
    pub vectors: Vec<f32>,     // n * dim
    pub neighbors: Vec<u32>,   // n * max_degree (padded with u32::MAX)
    pub neighbor_counts: Vec<u32>, // n
    pub entry: u32,
}

impl FlatGraph {
    #[inline]
    pub fn vec(&self, id: u32) -> &[f32] {
        let i = id as usize * self.dim;
        &self.vectors[i..i + self.dim]
    }
    #[inline]
    pub fn adj(&self, id: u32) -> &[u32] {
        let i = id as usize * self.max_degree;
        let k = self.neighbor_counts[id as usize] as usize;
        &self.neighbors[i..i + k]
    }
}

/// Ordered f32 wrapper for BinaryHeap.
#[derive(Copy, Clone, Debug)]
struct FOrd(f32, u32);
impl PartialEq for FOrd { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for FOrd {}
impl PartialOrd for FOrd {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for FOrd {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Beam search returning top-k by distance. `ef` is the candidate-list size.
pub fn search(g: &FlatGraph, q: &[f32], k: usize, ef: usize) -> Vec<(f32, u32)> {
    let mut visited = vec![false; g.n];
    // top-heap: max-heap of best-so-far (we pop worst on overflow)
    let mut top: BinaryHeap<FOrd> = BinaryHeap::with_capacity(ef + 1);
    // cand: min-heap by distance; std BinaryHeap is max, so wrap with negation
    let mut cand: BinaryHeap<std::cmp::Reverse<FOrd>> = BinaryHeap::with_capacity(ef + 1);

    let e = g.entry;
    let d = sq_l2(g.vec(e), q);
    visited[e as usize] = true;
    top.push(FOrd(d, e));
    cand.push(std::cmp::Reverse(FOrd(d, e)));

    while let Some(std::cmp::Reverse(c)) = cand.pop() {
        // stop condition: current candidate worse than worst in top
        if let Some(worst) = top.peek() {
            if c.0 > worst.0 && top.len() >= ef { break; }
        }
        for &nb in g.adj(c.1) {
            let ns = nb as usize;
            if ns >= g.n || visited[ns] { continue; }
            visited[ns] = true;
            let dn = sq_l2(g.vec(nb), q);
            let worst = top.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
            if top.len() < ef || dn < worst {
                top.push(FOrd(dn, nb));
                cand.push(std::cmp::Reverse(FOrd(dn, nb)));
                if top.len() > ef { top.pop(); }
            }
        }
    }
    let mut out: Vec<(f32, u32)> = top.into_iter().map(|f| (f.0, f.1)).collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(k);
    out
}

/// Ground-truth brute-force top-k (for recall measurement).
pub fn brute_topk(vectors: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = vectors.len() / dim;
    let mut heap: BinaryHeap<FOrd> = BinaryHeap::with_capacity(k + 1);
    for i in 0..n {
        let d = sq_l2(&vectors[i * dim..(i + 1) * dim], q);
        if heap.len() < k { heap.push(FOrd(d, i as u32)); }
        else if d < heap.peek().unwrap().0 {
            heap.pop();
            heap.push(FOrd(d, i as u32));
        }
    }
    let mut v: Vec<_> = heap.into_iter().collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().map(|x| x.1).collect()
}

/// Build a simple navigable graph:
///   1. Random pool of `pool` candidates per node (approx-NN warmup)
///   2. Keep `max_degree` nearest by L2
///   3. Symmetrise (add reverse edges up to max_degree)
/// Deterministic given the RNG seed.
pub fn build_graph(
    vectors: Vec<f32>,
    dim: usize,
    max_degree: usize,
    pool: usize,
    seed: u64,
) -> FlatGraph {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand::Rng;
    let n = vectors.len() / dim;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut neighbors = vec![u32::MAX; n * max_degree];
    let mut counts = vec![0u32; n];

    // Pass 1: per-node top-max_degree from a random pool.
    let mut pool_ids: Vec<u32> = Vec::with_capacity(pool);
    for i in 0..n {
        pool_ids.clear();
        for _ in 0..pool {
            let j = rng.gen_range(0..n as u32);
            if j != i as u32 { pool_ids.push(j); }
        }
        // score + partial sort
        let mut scored: Vec<(f32, u32)> = pool_ids.iter().map(|&j| {
            (sq_l2(&vectors[i*dim..(i+1)*dim], &vectors[j as usize *dim..(j as usize+1)*dim]), j)
        }).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.dedup_by_key(|x| x.1);
        let take = scored.len().min(max_degree);
        for (slot, &(_, j)) in scored.iter().take(take).enumerate() {
            neighbors[i * max_degree + slot] = j;
        }
        counts[i] = take as u32;
    }

    // Pass 2: symmetrise. For each edge i->j, ensure j->i if there is room.
    for i in 0..n {
        let cur_i_count = counts[i] as usize;
        for slot in 0..cur_i_count {
            let j = neighbors[i * max_degree + slot];
            if j == u32::MAX { continue; }
            let js = j as usize;
            let jc = counts[js] as usize;
            if jc >= max_degree { continue; }
            let base = js * max_degree;
            let already = (0..jc).any(|s| neighbors[base + s] == i as u32);
            if already { continue; }
            neighbors[base + jc] = i as u32;
            counts[js] = (jc + 1) as u32;
        }
    }

    // Entry point: medoid-ish (node with smallest average dist to 32 random samples).
    let sample: Vec<u32> = (0..32).map(|_| rng.gen_range(0..n as u32)).collect();
    let mut best = (f32::INFINITY, 0u32);
    for i in 0..n {
        let mut s = 0.0f32;
        for &j in &sample {
            s += sq_l2(&vectors[i*dim..(i+1)*dim], &vectors[j as usize*dim..(j as usize+1)*dim]);
        }
        if s < best.0 { best = (s, i as u32); }
    }

    FlatGraph {
        dim,
        max_degree,
        n,
        vectors,
        neighbors,
        neighbor_counts: counts,
        entry: best.1,
    }
}

/// Trait for pluggable node ordering strategies.
pub trait NodeOrdering {
    /// Returns permutation `old_to_new[old_id] = new_id`.
    fn permute(&self, g: &FlatGraph) -> Vec<u32>;
    fn name(&self) -> &'static str;
}

/// Apply a permutation to a FlatGraph, returning a re-ordered clone.
pub fn apply_permutation(g: &FlatGraph, old_to_new: &[u32]) -> FlatGraph {
    let n = g.n;
    let dim = g.dim;
    let md = g.max_degree;
    assert_eq!(old_to_new.len(), n);

    let mut new_to_old = vec![0u32; n];
    for (o, &nw) in old_to_new.iter().enumerate() { new_to_old[nw as usize] = o as u32; }

    let mut vectors = vec![0.0f32; n * dim];
    let mut neighbors = vec![u32::MAX; n * md];
    let mut counts = vec![0u32; n];

    for new_id in 0..n {
        let old_id = new_to_old[new_id] as usize;
        vectors[new_id*dim..(new_id+1)*dim]
            .copy_from_slice(&g.vectors[old_id*dim..(old_id+1)*dim]);
        let c = g.neighbor_counts[old_id] as usize;
        counts[new_id] = c as u32;
        for s in 0..c {
            let old_nb = g.neighbors[old_id * md + s];
            neighbors[new_id * md + s] = old_to_new[old_nb as usize];
        }
    }

    FlatGraph {
        dim, max_degree: md, n,
        vectors, neighbors,
        neighbor_counts: counts,
        entry: old_to_new[g.entry as usize],
    }
}

/// Convenience wrapper: build ordering, apply, return new graph.
pub fn reorder_with<O: NodeOrdering>(g: &FlatGraph, ord: &O) -> FlatGraph {
    let perm = ord.permute(g);
    apply_permutation(g, &perm)
}

/// Compute recall@k for a set of queries.
pub fn recall_at_k(
    approx: &[Vec<(f32, u32)>],
    truth: &[Vec<u32>],
    k: usize,
) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (a, t) in approx.iter().zip(truth.iter()) {
        let ts: std::collections::HashSet<u32> = t.iter().take(k).copied().collect();
        for (_, id) in a.iter().take(k) {
            if ts.contains(id) { hits += 1; }
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { hits as f32 / total as f32 }
}

/// Compute the average edge span (|new_id(u) - new_id(v)|) as a proxy for
/// how cache-local sequential neighbor probing will be.
pub fn mean_edge_span(g: &FlatGraph) -> f64 {
    let mut sum: u64 = 0;
    let mut cnt: u64 = 0;
    for i in 0..g.n {
        let c = g.neighbor_counts[i] as usize;
        for s in 0..c {
            let j = g.neighbors[i * g.max_degree + s] as i64;
            sum += (i as i64 - j).unsigned_abs();
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum as f64 / cnt as f64 }
}
