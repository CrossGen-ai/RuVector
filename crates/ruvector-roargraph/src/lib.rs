//! RoarGraph — projected bipartite graph index for cross-modal / OOD ANN.
//!
//! Reference: Chen et al., "RoarGraph: A Projected Bipartite Graph for Efficient
//! Cross-Modal Approximate Nearest Neighbor Search", VLDB 2024 (PVLDB 17.11).
//!
//! Algorithm summary (NAP — Neighborhood-Aware Projection):
//!   1. Given base set `B` and a training-query set `Q` drawn from the *query*
//!      distribution (e.g. text embeddings when `B` is image embeddings),
//!      compute the `k`-nearest base neighbours of every `q ∈ Q`.
//!   2. Build a bipartite graph with edges `q -> (its k base neighbours)` and
//!      project it onto `B`: any two base points that share a training query as
//!      a common neighbour become adjacency candidates with weight equal to the
//!      number of co-occurrences.
//!   3. RobustPrune the per-node adjacency to a bounded degree `M` using the
//!      DiskANN / Vamana alpha-pruning rule, which keeps a diverse set rather
//!      than only the closest by raw distance.
//!   4. Add reverse links and connect unreachable nodes to the medoid via beam
//!      search on the partially-built graph.
//!   5. Search is best-first beam search with a visited set.
//!
//! The central empirical claim — verified by `src/bin/bench.rs` on synthetic
//! OOD data — is that for *out-of-distribution* queries an in-base kNN graph
//! degrades sharply while a graph built from query-side training data keeps
//! high recall at the same memory footprint.

#![allow(clippy::needless_range_loop)]

pub mod distance;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub use distance::{l2_sq, Vector};

/// Build configuration.
#[derive(Debug, Clone)]
pub struct RoarConfig {
    /// k nearest base neighbours per training query (the bipartite degree).
    pub nap_k: usize,
    /// Max adjacency degree after RobustPrune.
    pub degree: usize,
    /// Beam width used during the supplementary connectivity pass.
    pub build_ef: usize,
    /// RobustPrune diversity factor (α ≥ 1.0). 1.2 matches DiskANN defaults.
    pub alpha: f32,
    /// Seed for any internal randomness (entry-point sampling).
    pub seed: u64,
}

impl Default for RoarConfig {
    fn default() -> Self {
        Self {
            nap_k: 32,
            degree: 32,
            build_ef: 64,
            alpha: 1.2,
            seed: 0xC0FFEE,
        }
    }
}

/// Ordered float wrapper for BinaryHeap (min-heap via Reverse).
#[derive(Copy, Clone, Debug)]
struct DistNode {
    dist: f32,
    id: u32,
}
impl PartialEq for DistNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for DistNode {}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap by distance, tiebreak by id for determinism
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}

/// In-memory ANN index. Owns the base vectors and the adjacency lists.
#[derive(Debug)]
pub struct RoarGraph {
    dim: usize,
    base: Vec<Vector>,
    adj: Vec<Vec<u32>>,
    entry: u32,
    cfg: RoarConfig,
}

impl RoarGraph {
    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.base.len()
    }
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn config(&self) -> &RoarConfig {
        &self.cfg
    }
    /// Total directed-edge count (sum of out-degrees).
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|n| n.len()).sum()
    }
    /// Average out-degree.
    pub fn avg_degree(&self) -> f32 {
        if self.adj.is_empty() {
            0.0
        } else {
            self.edge_count() as f32 / self.adj.len() as f32
        }
    }

    /// Build a RoarGraph from base vectors `base` using training queries `train_queries`.
    /// If `train_queries` is empty the index falls back to an in-base kNN graph,
    /// matching the standard NSG/Vamana initialisation.
    pub fn build(base: Vec<Vector>, train_queries: &[Vector], cfg: RoarConfig) -> Self {
        assert!(!base.is_empty(), "base set must be non-empty");
        let dim = base[0].len();
        assert!(base.iter().all(|v| v.len() == dim), "ragged base vectors");
        assert!(
            train_queries.iter().all(|v| v.len() == dim),
            "training query dim must match base dim"
        );

        let n = base.len();
        let nap_k = cfg.nap_k.min(n.saturating_sub(1)).max(1);
        let degree = cfg.degree.min(n.saturating_sub(1)).max(1);

        // Step 1+2: NAP. For each training query, find its top-nap_k base
        // neighbours; for each base node, accumulate its co-occurrence
        // neighbours weighted by frequency.
        let mut adj_cands: Vec<Vec<(u32, u32)>> = (0..n).map(|_| Vec::new()).collect();
        if !train_queries.is_empty() {
            let projected: Vec<Vec<u32>> = train_queries
                .par_iter()
                .map(|q| brute_knn(&base, q, nap_k))
                .collect();
            for q_neighbours in &projected {
                // for each unordered pair in q_neighbours -> add weight 1
                for i in 0..q_neighbours.len() {
                    for j in (i + 1)..q_neighbours.len() {
                        let (a, b) = (q_neighbours[i], q_neighbours[j]);
                        adj_cands[a as usize].push((b, 1));
                        adj_cands[b as usize].push((a, 1));
                    }
                }
            }
        }

        // Step 1b: if no training queries, seed adjacency with in-base kNN.
        if train_queries.is_empty() {
            let knn: Vec<Vec<u32>> = (0..n as u32)
                .collect::<Vec<_>>()
                .par_iter()
                .map(|&i| {
                    let mut ns = brute_knn(&base, &base[i as usize], nap_k + 1);
                    ns.retain(|&x| x != i);
                    ns.truncate(nap_k);
                    ns
                })
                .collect();
            for (i, ns) in knn.into_iter().enumerate() {
                for nb in ns {
                    adj_cands[i].push((nb, 1));
                }
            }
        }

        // Coalesce duplicate (id, weight) pairs into (id, total_weight).
        let coalesced: Vec<Vec<(u32, u32)>> = adj_cands
            .into_par_iter()
            .map(|mut v| {
                v.sort_unstable_by_key(|&(id, _)| id);
                let mut out: Vec<(u32, u32)> = Vec::with_capacity(v.len());
                for (id, w) in v {
                    if let Some(last) = out.last_mut() {
                        if last.0 == id {
                            last.1 += w;
                            continue;
                        }
                    }
                    out.push((id, w));
                }
                out
            })
            .collect();

        // Step 3: RobustPrune each adjacency.
        let mut adj: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                // Convert (id, weight) -> sorted-by-distance candidate list.
                // RoarGraph paper: NAP candidates sorted by *distance* before pruning,
                // because high co-occurrence already implies proximity in *query*
                // space but RobustPrune operates in base-space metric.
                let mut cands: Vec<(u32, f32)> = coalesced[i]
                    .iter()
                    .filter(|&&(id, _)| id != i as u32)
                    .map(|&(id, _w)| (id, l2_sq(&base[i], &base[id as usize])))
                    .collect();
                cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                robust_prune(&base, i as u32, cands, degree, cfg.alpha)
            })
            .collect();

        // Step 4: reverse-link augmentation — for every edge i -> j ensure
        // j -> i exists as a candidate, then re-prune.
        let mut rev: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..n {
            for &j in &adj[i] {
                if (j as usize) < n {
                    rev[j as usize].push(i as u32);
                }
            }
        }
        for i in 0..n {
            if rev[i].is_empty() {
                continue;
            }
            let mut merged: Vec<(u32, f32)> = adj[i]
                .iter()
                .chain(rev[i].iter())
                .copied()
                .filter(|&id| id != i as u32)
                .map(|id| (id, l2_sq(&base[i], &base[id as usize])))
                .collect();
            merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            merged.dedup_by_key(|x| x.0);
            adj[i] = robust_prune(&base, i as u32, merged, degree, cfg.alpha);
        }

        // Step 5: choose entry as the medoid (point with minimum sum of squared
        // distances to a small random sample).
        let entry = pick_medoid(&base, cfg.seed);

        // Step 6: connect unreachable nodes back via beam-search insertion.
        let mut g = RoarGraph {
            dim,
            base,
            adj,
            entry,
            cfg: cfg.clone(),
        };
        g.reachability_repair();
        g
    }

    fn reachability_repair(&mut self) {
        let n = self.base.len();
        let mut reached = vec![false; n];
        let mut stack = vec![self.entry as usize];
        reached[self.entry as usize] = true;
        while let Some(u) = stack.pop() {
            for &v in &self.adj[u] {
                let v = v as usize;
                if !reached[v] {
                    reached[v] = true;
                    stack.push(v);
                }
            }
        }
        let degree = self.cfg.degree;
        let alpha = self.cfg.alpha;
        let ef = self.cfg.build_ef.max(degree);
        for i in 0..n {
            if reached[i] {
                continue;
            }
            // Find nearest reached neighbours via beam search from entry.
            let cands_ids = self.beam_search_internal(&self.base[i].clone(), ef, i as u32);
            let mut cands: Vec<(u32, f32)> = cands_ids
                .into_iter()
                .filter(|&(id, _)| id != i as u32)
                .collect();
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let pruned = robust_prune(&self.base, i as u32, cands, degree, alpha);
            for &nb in &pruned {
                // back-link with bounded degree
                if !self.adj[nb as usize].contains(&(i as u32)) {
                    self.adj[nb as usize].push(i as u32);
                    if self.adj[nb as usize].len() > degree {
                        let merged: Vec<(u32, f32)> = self.adj[nb as usize]
                            .iter()
                            .copied()
                            .map(|id| (id, l2_sq(&self.base[nb as usize], &self.base[id as usize])))
                            .collect();
                        let mut sorted = merged;
                        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        self.adj[nb as usize] =
                            robust_prune(&self.base, nb, sorted, degree, alpha);
                    }
                }
            }
            self.adj[i] = pruned;
            reached[i] = true;
        }
    }

    /// Best-first beam search returning the top-`k` neighbours.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let scored = self.beam_search_internal(query, ef, u32::MAX);
        let mut out = scored;
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }

    /// Returns up to `ef` (id, dist²) pairs visited during best-first search.
    /// `skip_id == u32::MAX` disables skipping; otherwise the given id is not
    /// returned (used during repair so a node doesn't link to itself).
    fn beam_search_internal(&self, query: &[f32], ef: usize, skip_id: u32) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        let mut candidates: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::new();
        let mut results: BinaryHeap<DistNode> = BinaryHeap::new(); // max-heap (worst on top)

        let e = self.entry;
        let d0 = l2_sq(query, &self.base[e as usize]);
        visited.insert(e);
        candidates.push(std::cmp::Reverse(DistNode { dist: d0, id: e }));
        if e != skip_id {
            results.push(DistNode { dist: d0, id: e });
        }

        while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= ef && cur.dist > worst.dist {
                    break;
                }
            }
            for &nb in &self.adj[cur.id as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = l2_sq(query, &self.base[nb as usize]);
                let push_result = nb != skip_id
                    && (results.len() < ef || results.peek().map(|w| d < w.dist).unwrap_or(true));
                if push_result {
                    results.push(DistNode { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                    candidates.push(std::cmp::Reverse(DistNode { dist: d, id: nb }));
                } else if results.len() < ef {
                    candidates.push(std::cmp::Reverse(DistNode { dist: d, id: nb }));
                }
            }
        }
        results.into_iter().map(|n| (n.id, n.dist)).collect()
    }
}

/// Brute-force top-k base ids for `query`.
pub fn brute_knn(base: &[Vector], query: &[f32], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<DistNode> = BinaryHeap::with_capacity(k + 1);
    for (i, b) in base.iter().enumerate() {
        let d = l2_sq(query, b);
        if heap.len() < k {
            heap.push(DistNode { dist: d, id: i as u32 });
        } else if let Some(top) = heap.peek() {
            if d < top.dist {
                heap.pop();
                heap.push(DistNode { dist: d, id: i as u32 });
            }
        }
    }
    let mut v: Vec<DistNode> = heap.into_iter().collect();
    v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    v.into_iter().map(|n| n.id).collect()
}

/// DiskANN-style RobustPrune. Candidates must be sorted by ascending distance
/// to the point being pruned. Returns up to `degree` neighbour ids.
fn robust_prune(
    base: &[Vector],
    p: u32,
    mut cands: Vec<(u32, f32)>,
    degree: usize,
    alpha: f32,
) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(degree);
    while !cands.is_empty() && out.len() < degree {
        let (q, _dpq) = cands.remove(0);
        if q == p {
            continue;
        }
        out.push(q);
        // DiskANN α-RobustPrune: drop r when q occludes it, i.e. when
        // α · d(q, r) ≤ d(p, r). Keep r otherwise.
        let q_vec = &base[q as usize];
        cands.retain(|&(r, dpr)| {
            if r == q {
                return false;
            }
            let dqr = l2_sq(q_vec, &base[r as usize]);
            alpha * dqr > dpr
        });
    }
    out
}

/// Pick an approximate medoid: sample up to 256 random points, choose the one
/// with the smallest sum of squared L2 distances to the same sample.
fn pick_medoid(base: &[Vector], seed: u64) -> u32 {
    let n = base.len();
    if n == 1 {
        return 0;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let sample_size = 256.min(n);
    let mut ids: Vec<u32> = (0..n as u32).collect();
    ids.shuffle(&mut rng);
    let sample: Vec<u32> = ids.into_iter().take(sample_size).collect();
    let scored: Vec<(u32, f32)> = sample
        .par_iter()
        .map(|&i| {
            let v = &base[i as usize];
            let s: f32 = sample.iter().map(|&j| l2_sq(v, &base[j as usize])).sum();
            (i, s)
        })
        .collect();
    scored
        .into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|x| x.0)
        .unwrap_or(0)
}

// ---------- baselines used by the benchmark binary ----------

/// In-base kNN graph + beam search. The classic "ID-only" Vamana init.
#[derive(Debug)]
pub struct KnnGraph {
    inner: RoarGraph,
}

impl KnnGraph {
    pub fn build(base: Vec<Vector>, cfg: RoarConfig) -> Self {
        Self {
            inner: RoarGraph::build(base, &[], cfg),
        }
    }
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.inner.search(q, k, ef)
    }
    pub fn avg_degree(&self) -> f32 {
        self.inner.avg_degree()
    }
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
}

/// Random graph with bounded degree + beam search. Degenerate worst-case
/// baseline included for honest comparison.
#[derive(Debug)]
pub struct RandomGraph {
    base: Vec<Vector>,
    adj: Vec<Vec<u32>>,
    entry: u32,
}

impl RandomGraph {
    pub fn build(base: Vec<Vector>, degree: usize, seed: u64) -> Self {
        let n = base.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut adj: Vec<Vec<u32>> = Vec::with_capacity(n);
        let degree = degree.min(n.saturating_sub(1)).max(1);
        for i in 0..n {
            let mut nbs: Vec<u32> = Vec::with_capacity(degree);
            while nbs.len() < degree {
                let j: u32 = rng.gen_range(0..n as u32);
                if j as usize != i && !nbs.contains(&j) {
                    nbs.push(j);
                }
            }
            adj.push(nbs);
        }
        let entry = pick_medoid(&base, seed);
        Self { base, adj, entry }
    }
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|n| n.len()).sum()
    }
    pub fn avg_degree(&self) -> f32 {
        if self.adj.is_empty() {
            0.0
        } else {
            self.edge_count() as f32 / self.adj.len() as f32
        }
    }
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let n = self.base.len();
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        let mut candidates: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::new();
        let mut results: BinaryHeap<DistNode> = BinaryHeap::new();
        let e = self.entry;
        let d0 = l2_sq(query, &self.base[e as usize]);
        visited.insert(e);
        candidates.push(std::cmp::Reverse(DistNode { dist: d0, id: e }));
        results.push(DistNode { dist: d0, id: e });
        while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= ef && cur.dist > worst.dist {
                    break;
                }
            }
            for &nb in &self.adj[cur.id as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = l2_sq(query, &self.base[nb as usize]);
                if results.len() < ef || results.peek().map(|w| d < w.dist).unwrap_or(true) {
                    results.push(DistNode { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                    candidates.push(std::cmp::Reverse(DistNode { dist: d, id: nb }));
                }
            }
            if visited.len() >= n {
                break;
            }
        }
        let mut out: Vec<(u32, f32)> = results.into_iter().map(|n| (n.id, n.dist)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    fn gen(n: usize, d: usize, mean: f32, seed: u64) -> Vec<Vector> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>() + mean).collect())
            .collect()
    }

    fn recall(got: &[(u32, f32)], truth: &[u32]) -> f32 {
        let truth: HashSet<u32> = truth.iter().copied().collect();
        let hits = got.iter().filter(|(id, _)| truth.contains(id)).count();
        hits as f32 / truth.len().max(1) as f32
    }

    #[test]
    fn roar_beats_knn_on_ood() {
        let base = gen(800, 16, 0.0, 1);
        let train = gen(400, 16, 1.5, 2); // shifted distribution
        let queries = gen(50, 16, 1.5, 3); // same shift as train -> OOD vs base
        let k = 10;
        let ef = 40;

        let truth: Vec<Vec<u32>> = queries.iter().map(|q| brute_knn(&base, q, k)).collect();

        let cfg = RoarConfig { nap_k: 16, degree: 16, build_ef: 32, alpha: 1.2, seed: 1 };
        let roar = RoarGraph::build(base.clone(), &train, cfg.clone());
        let knn = KnnGraph::build(base.clone(), cfg.clone());

        let r_roar: f32 = queries
            .iter()
            .zip(truth.iter())
            .map(|(q, t)| recall(&roar.search(q, k, ef), t))
            .sum::<f32>()
            / queries.len() as f32;
        let r_knn: f32 = queries
            .iter()
            .zip(truth.iter())
            .map(|(q, t)| recall(&knn.search(q, k, ef), t))
            .sum::<f32>()
            / queries.len() as f32;

        println!("OOD recall@{k}: roar={r_roar:.3}  knn={r_knn:.3}");
        assert!(r_roar >= r_knn - 0.02, "roar should match or beat knn on OOD");
        assert!(r_roar >= 0.7, "roar OOD recall should be high, got {r_roar:.3}");
    }

    #[test]
    fn random_graph_is_a_floor() {
        let base = gen(400, 8, 0.0, 4);
        let queries = gen(20, 8, 0.0, 5);
        let truth: Vec<Vec<u32>> = queries.iter().map(|q| brute_knn(&base, q, 5)).collect();
        let rg = RandomGraph::build(base.clone(), 16, 7);
        let r: f32 = queries
            .iter()
            .zip(truth.iter())
            .map(|(q, t)| recall(&rg.search(q, 5, 32), t))
            .sum::<f32>()
            / queries.len() as f32;
        println!("random-graph recall@5: {r:.3}");
        assert!(r > 0.1, "random graph should at least find something");
    }
}
