//! NSG build (MRNG pruning + DFS connectivity) and beam-search query.

use crate::error::{NsgError, Result};
use crate::knn::{centroid, nearest_to, nn_descent, Heap};
use crate::l2_sq;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
pub struct SearchHit {
    pub id: u32,
    /// Squared L2 distance.
    pub dist: f32,
}

impl PartialEq for SearchHit {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && self.dist == o.dist
    }
}
impl Eq for SearchHit {}

#[derive(Clone, Copy, Debug)]
struct OrdHit(SearchHit);

impl PartialEq for OrdHit {
    fn eq(&self, o: &Self) -> bool {
        self.0.dist == o.0.dist
    }
}
impl Eq for OrdHit {}
impl PartialOrd for OrdHit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdHit {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is max-heap; we want min-by-dist semantics for the
        // candidate frontier, so we *reverse* the comparison.
        o.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}

/// Tunable parameters for `NsgBuilder`.
#[derive(Clone, Copy, Debug)]
pub struct NsgParams {
    /// Out-degree cap (R in the NSG paper). Typical 20-50.
    pub r: usize,
    /// Candidate pool size used during MRNG selection (L in the paper). Must
    /// be ≥ `r`. Typical 40-100.
    pub l_build: usize,
    /// k for the kNN graph init (K in NN-Descent). Typical 50.
    pub k_knn: usize,
    /// NN-Descent sweep budget.
    pub knn_iters: usize,
    /// NN-Descent local-join sample rate (0.4-1.0).
    pub knn_sample: f32,
    /// Deterministic seed.
    pub seed: u64,
    /// Pruning relaxation factor α ≥ 1.0 (Vamana/DiskANN extension). The
    /// occlusion test becomes `α · d(s, cand) < d(p, cand)`, so larger α
    /// keeps more edges → denser graph, higher recall on tight clusters.
    /// `α = 1.0` recovers the strict NSG MRNG rule.
    pub alpha: f32,
}

impl Default for NsgParams {
    fn default() -> Self {
        Self {
            r: 32,
            l_build: 64,
            k_knn: 50,
            knn_iters: 6,
            knn_sample: 1.0,
            seed: 0xA17,
            alpha: 1.2,
        }
    }
}

pub struct NsgBuilder {
    params: NsgParams,
}

impl NsgBuilder {
    pub fn new(params: NsgParams) -> Self {
        Self { params }
    }

    pub fn build(self, data: Vec<Vec<f32>>) -> Result<NsgIndex> {
        let n = data.len();
        if n == 0 {
            return Err(NsgError::Empty);
        }
        if self.params.r == 0 {
            return Err(NsgError::BadParam { name: "r" });
        }
        if self.params.l_build < self.params.r {
            return Err(NsgError::BadParam { name: "l_build" });
        }
        if self.params.k_knn == 0 || self.params.k_knn >= n {
            return Err(NsgError::BadParam { name: "k_knn" });
        }
        let dim = data[0].len();
        for v in &data {
            if v.len() != dim {
                return Err(NsgError::DimMismatch {
                    index: dim,
                    query: v.len(),
                });
            }
        }

        // Step 1 — approximate kNN graph.
        let knn = nn_descent(
            &data,
            self.params.k_knn,
            self.params.knn_iters,
            self.params.knn_sample,
            self.params.seed,
        );

        // Step 2 — navigating node = base point closest to centroid.
        let c = centroid(&data);
        let nav = nearest_to(&data, &knn, &c, self.params.seed ^ 0x91);

        // Step 3 — MRNG-pruned adjacency from a per-node candidate pool.
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for p in 0..n {
            let pool = candidate_pool(&data, &knn, nav, p as u32, self.params.l_build);
            adj[p] = mrng_select(&data, &pool, p as u32, self.params.r, self.params.alpha);
        }

        // Step 3b — Reverse-edge insertion + per-node MRNG re-prune. Critical
        // for connectivity: without this, MRNG drops too many edges in dense
        // clusters and the search graph fragments (Fu et al. Alg. 2, step 6).
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for p in 0..n {
            for &q in &adj[p] {
                rev[q as usize].push(p as u32);
            }
        }
        for q in 0..n {
            if rev[q].is_empty() {
                continue;
            }
            // Union current out-edges with incoming reverse edges, sort by
            // d(q, .), then re-apply MRNG pruning with cap R.
            let mut union: Vec<SearchHit> = adj[q]
                .iter()
                .chain(rev[q].iter())
                .copied()
                .map(|id| SearchHit {
                    id,
                    dist: l2_sq(&data[q], &data[id as usize]),
                })
                .collect();
            // Dedup.
            union.sort_by(|a, b| a.id.cmp(&b.id));
            union.dedup_by_key(|h| h.id);
            union.retain(|h| h.id as usize != q);
            union.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            adj[q] = mrng_select(&data, &union, q as u32, self.params.r, self.params.alpha);
        }

        // Step 4 — DFS tree augmentation. Any vertex not reachable from nav
        // is wired into the spanning tree from its nearest reached ancestor.
        connect_dfs(&mut adj, nav, &data);

        Ok(NsgIndex {
            data,
            adj,
            nav,
            dim,
        })
    }
}

/// Run a greedy beam search from `nav` toward `p` on the kNN graph, returning
/// a candidate pool of up to `l_build` nodes (excluding `p` itself).
fn candidate_pool(
    data: &[Vec<f32>],
    knn: &[Heap],
    nav: u32,
    p: u32,
    l_build: usize,
) -> Vec<SearchHit> {
    let n = data.len();
    let target = &data[p as usize];
    let mut visited = vec![false; n];

    // Min-heap on distance for the active frontier.
    let mut frontier: BinaryHeap<OrdHit> = BinaryHeap::new();
    // Max-heap (default) bounded to `l_build` for the result pool.
    let mut pool: BinaryHeap<SearchHitMax> = BinaryHeap::new();

    let seed_d = l2_sq(&data[nav as usize], target);
    visited[nav as usize] = true;
    frontier.push(OrdHit(SearchHit { id: nav, dist: seed_d }));
    pool.push(SearchHitMax(SearchHit { id: nav, dist: seed_d }));

    // Pre-seed the frontier and pool with `p`'s own kNN entries. This is
    // critical on datasets with tight clusters where a pure nav-to-p greedy
    // walk never enters the right basin.  Equivalent to the §4.3 "candidate
    // pool = greedy_search(nav, p) ∪ knn[p]" set in the NSG paper.
    for e in &knn[p as usize].items {
        if e.id == p || visited[e.id as usize] {
            continue;
        }
        visited[e.id as usize] = true;
        let d = l2_sq(&data[e.id as usize], target);
        frontier.push(OrdHit(SearchHit { id: e.id, dist: d }));
        pool.push(SearchHitMax(SearchHit { id: e.id, dist: d }));
        if pool.len() > l_build {
            pool.pop();
        }
    }

    while let Some(OrdHit(top)) = frontier.pop() {
        // Stop if even the closest unexpanded is worse than worst in pool.
        if pool.len() >= l_build {
            if top.dist >= pool.peek().unwrap().0.dist {
                break;
            }
        }
        for &nb in knn[top.id as usize].items.iter().map(|e| &e.id) {
            if visited[nb as usize] || nb == p {
                continue;
            }
            visited[nb as usize] = true;
            let d = l2_sq(&data[nb as usize], target);
            frontier.push(OrdHit(SearchHit { id: nb, dist: d }));
            pool.push(SearchHitMax(SearchHit { id: nb, dist: d }));
            if pool.len() > l_build {
                pool.pop();
            }
        }
    }

    let mut out: Vec<SearchHit> = pool.into_iter().map(|h| h.0).collect();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    out
}

/// Apply the MRNG occlusion rule to a sorted-by-distance candidate pool.
fn mrng_select(data: &[Vec<f32>], pool: &[SearchHit], p: u32, r: usize, alpha: f32) -> Vec<u32> {
    // NSG Algorithm 2 — `pool` is sorted by ascending d(p, cand), so any
    // already-`keep`ed selected `s` is automatically closer to `p` than the
    // current `cand`.  Therefore the MRNG occlusion test reduces to
    // `d(s, cand) < d(p, cand)` (Fu et al. §4.3, Definition 6).
    let _ = data; // silence unused-import lint on identical-name file
    let mut keep: Vec<u32> = Vec::with_capacity(r);
    for cand in pool {
        if cand.id == p {
            continue;
        }
        let dp_c = cand.dist;
        let mut occluded = false;
        for &s in &keep {
            let d_sc = l2_sq(&data[s as usize], &data[cand.id as usize]);
            if alpha * d_sc < dp_c {
                occluded = true;
                break;
            }
        }
        if !occluded {
            keep.push(cand.id);
            if keep.len() == r {
                break;
            }
        }
    }
    keep
}

/// DFS from `nav`; for every unreached vertex `v`, add the edge
/// `nearest_reached(v) → v` so a monotonic path from `nav` to `v` exists.
fn connect_dfs(adj: &mut [Vec<u32>], nav: u32, data: &[Vec<f32>]) {
    let n = adj.len();
    let mut reached = vec![false; n];
    let mut stack: Vec<u32> = Vec::with_capacity(n);
    stack.push(nav);
    reached[nav as usize] = true;
    while let Some(v) = stack.pop() {
        for nb in adj[v as usize].clone() {
            if !reached[nb as usize] {
                reached[nb as usize] = true;
                stack.push(nb);
            }
        }
    }
    // Snapshot the reached set to scan for a sponsor.
    let mut reached_ids: Vec<u32> = (0..n as u32).filter(|&i| reached[i as usize]).collect();
    let mut to_add: Vec<(u32, u32)> = Vec::new();
    for v in 0..n as u32 {
        if reached[v as usize] {
            continue;
        }
        // Find closest reached vertex.
        let mut best = reached_ids[0];
        let mut best_d = l2_sq(&data[best as usize], &data[v as usize]);
        for &r in &reached_ids[1..] {
            let d = l2_sq(&data[r as usize], &data[v as usize]);
            if d < best_d {
                best = r;
                best_d = d;
            }
        }
        to_add.push((best, v));
        reached[v as usize] = true;
        reached_ids.push(v);
    }
    for (s, t) in to_add {
        adj[s as usize].push(t);
    }
}

pub struct NsgIndex {
    data: Vec<Vec<f32>>,
    adj: Vec<Vec<u32>>,
    nav: u32,
    dim: usize,
}

impl NsgIndex {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn navigating_node(&self) -> u32 {
        self.nav
    }

    /// Memory footprint of the graph (edges) in bytes — excludes vector data.
    pub fn graph_bytes(&self) -> usize {
        self.adj.iter().map(|v| v.capacity() * 4).sum::<usize>()
            + self.adj.capacity() * std::mem::size_of::<Vec<u32>>()
    }

    /// Average out-degree.
    pub fn avg_degree(&self) -> f32 {
        let total: usize = self.adj.iter().map(|v| v.len()).sum();
        total as f32 / self.adj.len() as f32
    }

    /// Search for `k` nearest neighbors using beam width `l_search`.
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Result<Vec<SearchHit>> {
        if query.len() != self.dim {
            return Err(NsgError::DimMismatch {
                index: self.dim,
                query: query.len(),
            });
        }
        if k == 0 || k > self.data.len() {
            return Err(NsgError::BadK { k, n: self.data.len() });
        }
        let l = l_search.max(k);

        let n = self.data.len();
        let mut visited = vec![false; n];
        let mut frontier: BinaryHeap<OrdHit> = BinaryHeap::new();
        let mut pool: BinaryHeap<SearchHitMax> = BinaryHeap::new();

        let seed_d = l2_sq(&self.data[self.nav as usize], query);
        visited[self.nav as usize] = true;
        frontier.push(OrdHit(SearchHit { id: self.nav, dist: seed_d }));
        pool.push(SearchHitMax(SearchHit { id: self.nav, dist: seed_d }));

        while let Some(OrdHit(top)) = frontier.pop() {
            if pool.len() >= l && top.dist >= pool.peek().unwrap().0.dist {
                break;
            }
            for &nb in &self.adj[top.id as usize] {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let d = l2_sq(&self.data[nb as usize], query);
                frontier.push(OrdHit(SearchHit { id: nb, dist: d }));
                pool.push(SearchHitMax(SearchHit { id: nb, dist: d }));
                if pool.len() > l {
                    pool.pop();
                }
            }
        }

        let mut out: Vec<SearchHit> = pool.into_iter().map(|h| h.0).collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out.truncate(k);
        Ok(out)
    }
}

/// Max-heap wrapper (sorts by `dist` ascending → larger pops first).
#[derive(Clone, Copy)]
struct SearchHitMax(SearchHit);
impl PartialEq for SearchHitMax {
    fn eq(&self, o: &Self) -> bool {
        self.0.dist == o.0.dist
    }
}
impl Eq for SearchHitMax {}
impl PartialOrd for SearchHitMax {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for SearchHitMax {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.dist.partial_cmp(&o.0.dist).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{brute_force_topk, recall_at_k};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn gauss_clusters(n: usize, d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        // Use overlapping clusters — realistic ANN workload, not isolated dots.
        let centers: Vec<Vec<f32>> = (0..c)
            .map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect())
            .collect();
        (0..n)
            .map(|_| {
                let cc = &centers[rng.gen_range(0..c)];
                cc.iter().map(|x| x + rng.gen_range(-1.5..1.5)).collect()
            })
            .collect()
    }

    #[test]
    fn build_small_and_search() {
        let data = gauss_clusters(500, 16, 8, 7);
        let queries = gauss_clusters(50, 16, 8, 8);

        let index = NsgBuilder::new(NsgParams {
            r: 24,
            l_build: 80,
            k_knn: 40,
            knn_iters: 6,
            knn_sample: 1.0,
            seed: 11,
            alpha: 1.2,
        })
        .build(data.clone())
        .expect("build");

        eprintln!("avg_degree = {:.3}", index.avg_degree());
        assert!(index.avg_degree() > 1.0);

        let mut total = 0.0f32;
        for q in &queries {
            let gt = brute_force_topk(&data, q, 10);
            let pred = index.search(q, 10, 40).unwrap();
            total += recall_at_k(&gt, &pred);
        }
        let recall = total / queries.len() as f32;
        eprintln!("recall@10 = {:.4}", recall);
        assert!(recall > 0.8, "recall@10 = {recall}");
    }

    #[test]
    fn errors_on_bad_inputs() {
        let r = NsgBuilder::new(NsgParams::default()).build(Vec::new());
        assert!(matches!(r, Err(NsgError::Empty)));
    }
}
