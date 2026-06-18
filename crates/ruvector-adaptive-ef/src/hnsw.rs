//! A small, self-contained HNSW search kernel.
//!
//! This is intentionally minimal: a single-layer NSW graph (i.e. the
//! base layer of HNSW) with bounded neighbour lists.  The point of this
//! crate is to study the `ef`-vs-recall tradeoff, not to ship a
//! best-of-breed HNSW — for that, see `crates/ruvector-core`.
//!
//! What you get from this kernel that matters for the experiment:
//! * Configurable `ef` per search (the knob we want to adapt).
//! * An exact `distance_evaluations` counter — the canonical proxy for
//!   ANN search cost (it tracks linearly with wall time and is invariant
//!   to CPU noise, which matters for nightly CI runs).
//! * Reproducibility: graph construction is deterministic given a seed.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Squared L2 distance.  Inlined so the compiler can auto-vectorise.
#[inline(always)]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// A node in the max-heap used to track the current candidate frontier.
#[derive(Copy, Clone, Debug)]
struct Node {
    /// Distance from the query — float ordered via `total_cmp`.
    dist: f32,
    /// Index into the vector store.
    id: u32,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist.total_cmp(&other.dist).is_eq()
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // For a max-heap by distance.
        self.dist.total_cmp(&other.dist)
    }
}

/// Inverted node (min-heap helper).
#[derive(Copy, Clone, Debug)]
struct MinNode(Node);
impl PartialEq for MinNode {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MinNode {}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinNode {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.dist.total_cmp(&self.0.dist)
    }
}

/// Per-search instrumentation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    pub distance_evaluations: u64,
    pub hops: u64,
}

/// Minimal NSW base-layer index.
pub struct MiniHnsw {
    vectors: Vec<Vec<f32>>,
    /// Adjacency: `neighbours[i]` is the bounded list of out-edges of `i`.
    neighbours: Vec<Vec<u32>>,
    /// Default entry-point — fixed at construction time.
    entry: u32,
}

impl MiniHnsw {
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// k-NN search with configurable `ef` (priority queue size).
    ///
    /// Returns the top-`k` ids sorted by ascending distance, plus stats.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<(u32, f32)>, SearchStats) {
        self.search_from(query, k, ef, &[self.entry])
    }

    /// Like [`search`] but starts the traversal from the given seed
    /// node ids.  Useful during construction/refinement and for
    /// multi-start queries.
    pub fn search_from(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        seeds: &[u32],
    ) -> (Vec<(u32, f32)>, SearchStats) {
        let mut stats = SearchStats::default();
        let ef = ef.max(k).max(1);

        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        let mut frontier: BinaryHeap<MinNode> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<Node> = BinaryHeap::with_capacity(ef + 1);

        for &s in seeds {
            if visited.insert(s) {
                let d = squared_l2(query, &self.vectors[s as usize]);
                stats.distance_evaluations += 1;
                let n = Node { dist: d, id: s };
                frontier.push(MinNode(n));
                results.push(n);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(MinNode(cand)) = frontier.pop() {
            // Best result in the bounded heap is at the top (max-heap).
            let worst_result = results.peek().map(|n| n.dist).unwrap_or(f32::INFINITY);
            if cand.dist > worst_result && results.len() >= ef {
                break; // No remaining frontier candidate can improve top-ef.
            }
            stats.hops += 1;

            for &nb in &self.neighbours[cand.id as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = squared_l2(query, &self.vectors[nb as usize]);
                stats.distance_evaluations += 1;
                let worst = results.peek().map(|n| n.dist).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    let n = Node { dist: d, id: nb };
                    frontier.push(MinNode(n));
                    results.push(n);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results.into_iter().map(|n| (n.id, n.dist)).collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out.truncate(k);
        (out, stats)
    }

    /// Brute-force ground-truth k-NN — used for recall measurement.
    pub fn brute_force(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut all: Vec<(u32, f32)> = (0..self.vectors.len() as u32)
            .map(|i| (i, squared_l2(query, &self.vectors[i as usize])))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.truncate(k);
        all
    }
}

/// Greedy NSW graph builder: insert vectors one at a time, each new node
/// queries the current graph at small `ef` and keeps the closest `m`
/// neighbours.  This produces a good-enough small-world graph for our
/// experiment.  (We do NOT shipping-grade prune — see `ruvector-core`
/// for that.)
pub struct MiniHnswBuilder {
    dim: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
    refine_passes: usize,
}

impl MiniHnswBuilder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            m: 16,
            ef_construction: 64,
            seed: 0xCAFE_F00D,
            refine_passes: 1,
        }
    }
    pub fn refine_passes(mut self, p: usize) -> Self {
        self.refine_passes = p;
        self
    }
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Brute-force kNN build: for each node, set its `m` neighbours to the
    /// closest data points (excluding itself) found by exhaustive scan,
    /// then add `extra_random` long-range edges per node for
    /// small-world navigability.  This is O(N²·D) — only practical up
    /// to ~50K nodes — but it produces a *known-good* graph, which is
    /// what we want for this experiment.  The expensive scan is
    /// parallelised over rayon.
    pub fn build(self, vectors: Vec<Vec<f32>>) -> MiniHnsw {
        use rayon::prelude::*;
        assert!(!vectors.is_empty());
        assert!(vectors.iter().all(|v| v.len() == self.dim));
        let n = vectors.len();
        let m = self.m;
        // Split m: m_close for nearest-neighbours, m_rand for long-range.
        let m_rand = (m / 4).max(2);
        let m_close = m - m_rand;
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Brute-force m_close-NN for every node, in parallel.
        let neighbours_close: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vi = &vectors[i];
                let mut heap: BinaryHeap<Node> = BinaryHeap::with_capacity(m_close + 1);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let d = squared_l2(vi, &vectors[j]);
                    if heap.len() < m_close {
                        heap.push(Node { dist: d, id: j as u32 });
                    } else if d < heap.peek().unwrap().dist {
                        heap.push(Node { dist: d, id: j as u32 });
                        heap.pop();
                    }
                }
                heap.into_iter().map(|n| n.id).collect()
            })
            .collect();

        // Long-range random edges (small-world).
        let mut neighbours: Vec<Vec<u32>> = neighbours_close;
        for i in 0..n {
            let cur_set: HashSet<u32> = neighbours[i].iter().copied().collect();
            for _ in 0..m_rand {
                // Sample until we find a fresh id.
                for _ in 0..16 {
                    let r = rng.gen_range(0..n) as u32;
                    if r as usize != i && !cur_set.contains(&r) && !neighbours[i].contains(&r) {
                        neighbours[i].push(r);
                        break;
                    }
                }
            }
        }

        // Pick centroid-closest entry.
        let centroid: Vec<f32> = {
            let mut c = vec![0.0f32; self.dim];
            for v in &vectors {
                for (ci, x) in c.iter_mut().zip(v.iter()) {
                    *ci += *x;
                }
            }
            let inv = 1.0 / n as f32;
            for x in &mut c {
                *x *= inv;
            }
            c
        };
        let entry = (0..n as u32)
            .min_by(|&a, &b| {
                squared_l2(&vectors[a as usize], &centroid)
                    .total_cmp(&squared_l2(&vectors[b as usize], &centroid))
            })
            .unwrap();

        return MiniHnsw {
            vectors,
            neighbours,
            entry,
        };

        // Legacy greedy build kept below for reference; unreachable.
        #[allow(unreachable_code)]
        {

        // Pick a stable entry point near the centroid.
        let centroid: Vec<f32> = {
            let mut c = vec![0.0f32; self.dim];
            for v in &vectors {
                for (ci, x) in c.iter_mut().zip(v.iter()) {
                    *ci += *x;
                }
            }
            let inv = 1.0 / n as f32;
            for x in &mut c {
                *x *= inv;
            }
            c
        };
        let entry = (0..n as u32)
            .min_by(|&a, &b| {
                squared_l2(&vectors[a as usize], &centroid)
                    .total_cmp(&squared_l2(&vectors[b as usize], &centroid))
            })
            .unwrap();

        // Bootstrap: make the first `min(m, n-1)` non-entry nodes a small
        // mesh with `entry`.  This gives the partial graph a connected
        // seed.  After this, every newly-inserted node will find these
        // bootstrap nodes via graph search.
        let neighbours: Vec<Vec<u32>> = vec![Vec::with_capacity(m); n];
        let mut partial = MiniHnsw {
            vectors,
            neighbours,
            entry,
        };
        let bootstrap_size = m.min(n.saturating_sub(1));
        let mut bootstrap_ids: Vec<u32> = (0..n as u32).filter(|&i| i != entry).take(bootstrap_size).collect();
        bootstrap_ids.push(entry);
        for &i in &bootstrap_ids {
            for &j in &bootstrap_ids {
                if i != j && partial.neighbours[i as usize].len() < m {
                    partial.neighbours[i as usize].push(j);
                }
            }
        }

        // Deterministic random insertion order (excluding bootstrap nodes,
        // which are already wired).
        let bootstrap_set: HashSet<u32> = bootstrap_ids.iter().copied().collect();
        let mut order: Vec<usize> = (0..n).filter(|i| !bootstrap_set.contains(&(*i as u32))).collect();
        for i in (1..order.len()).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        for &i in &order {
            let q = partial.vectors[i].clone();
            let (nbrs, _) = partial.search(&q, m, self.ef_construction);
            let new_nbrs: Vec<u32> = nbrs
                .into_iter()
                .filter(|(id, _)| *id as usize != i)
                .map(|(id, _)| id)
                .take(m)
                .collect();
            partial.neighbours[i] = new_nbrs.clone();

            // Add reverse links — evict the FARTHEST existing neighbour
            // if the list is full.  Standard HNSW pruning policy.
            for &nb in &new_nbrs {
                if partial.neighbours[nb as usize].contains(&(i as u32)) {
                    continue;
                }
                if partial.neighbours[nb as usize].len() < m {
                    partial.neighbours[nb as usize].push(i as u32);
                } else {
                    // Evict farthest from `nb` if `i` is closer.
                    let d_new = squared_l2(&partial.vectors[nb as usize], &partial.vectors[i]);
                    let (farthest_idx, farthest_d) = partial.neighbours[nb as usize]
                        .iter()
                        .enumerate()
                        .map(|(k, &other)| {
                            (k, squared_l2(&partial.vectors[nb as usize], &partial.vectors[other as usize]))
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap();
                    if d_new < farthest_d {
                        partial.neighbours[nb as usize][farthest_idx] = i as u32;
                    }
                }
            }
        }

        // Refinement passes: re-search the now-fully-built graph for each
        // node and replace its neighbour list with the best `m` results.
        // We seed search from node `i` itself plus its current neighbours
        // and a small random sample — this protects against orphaned
        // nodes that aren't reachable from the global entry, and lets
        // every node's local neighbourhood improve to near-optimal.
        let n_random_seeds = 4usize;
        for _ in 0..self.refine_passes {
            let mut updated: Vec<Vec<u32>> = vec![Vec::with_capacity(m); n];
            for i in 0..n {
                let q = partial.vectors[i].clone();
                let mut seeds: Vec<u32> = Vec::with_capacity(1 + m + n_random_seeds);
                seeds.push(i as u32);
                seeds.extend(partial.neighbours[i].iter().copied());
                seeds.push(entry);
                for _ in 0..n_random_seeds {
                    seeds.push(rng.gen_range(0..n) as u32);
                }
                let (nbrs, _) = partial.search_from(
                    &q,
                    m + 1,
                    self.ef_construction.max(m * 4),
                    &seeds,
                );
                let new_nbrs: Vec<u32> = nbrs
                    .into_iter()
                    .filter(|(id, _)| *id as usize != i)
                    .map(|(id, _)| id)
                    .take(m)
                    .collect();
                updated[i] = new_nbrs;
            }
            partial.neighbours = updated;
            // Re-add symmetric edges (bounded, eviction by distance).
            for i in 0..n {
                let nbrs_i = partial.neighbours[i].clone();
                for nb in nbrs_i {
                    if partial.neighbours[nb as usize].contains(&(i as u32)) {
                        continue;
                    }
                    if partial.neighbours[nb as usize].len() < m {
                        partial.neighbours[nb as usize].push(i as u32);
                    } else {
                        let d_new = squared_l2(&partial.vectors[nb as usize], &partial.vectors[i]);
                        let (k_far, d_far) = partial.neighbours[nb as usize]
                            .iter()
                            .enumerate()
                            .map(|(k, &o)| {
                                (k, squared_l2(&partial.vectors[nb as usize], &partial.vectors[o as usize]))
                            })
                            .max_by(|a, b| a.1.total_cmp(&b.1))
                            .unwrap();
                        if d_new < d_far {
                            partial.neighbours[nb as usize][k_far] = i as u32;
                        }
                    }
                }
            }
        }

        // Connectivity repair: BFS from `entry`, any node not reached is
        // attached by APPENDING an incoming edge from the closest
        // reachable node (allowing the list to temporarily exceed `m`
        // — eviction here could orphan other nodes and produce an
        // infinite repair loop).  Single pass is enough because every
        // unreached node directly gains an inbound edge from a reached
        // node.
        let mut reached = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        q.push_back(entry);
        reached[entry as usize] = true;
        while let Some(u) = q.pop_front() {
            for &v in &partial.neighbours[u as usize] {
                if !reached[v as usize] {
                    reached[v as usize] = true;
                    q.push_back(v);
                }
            }
        }
        let reached_ids: Vec<u32> = (0..n as u32).filter(|&i| reached[i as usize]).collect();
        let unreached: Vec<u32> = (0..n as u32).filter(|&i| !reached[i as usize]).collect();
        let n_orphans = unreached.len();
        for &u in &unreached {
            let uv = &partial.vectors[u as usize];
            let (closest, _) = reached_ids
                .iter()
                .map(|&r| (r, squared_l2(&partial.vectors[r as usize], uv)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            // Append (no eviction) — list may briefly exceed m, which is
            // fine because search only iterates the list once.
            partial.neighbours[closest as usize].push(u);
        }
        if n_orphans > 0 {
            eprintln!(
                "[ruvector-adaptive-ef] repaired {} orphan nodes ({:.1}%)",
                n_orphans,
                100.0 * n_orphans as f32 / n as f32
            );
        }

        partial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn synthetic(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let nd = Normal::new(0.0f32, 1.0).unwrap();
        (0..n).map(|_| (0..dim).map(|_| nd.sample(&mut rng)).collect()).collect()
    }

    #[test]
    fn search_recall_grows_with_ef() {
        let data = synthetic(500, 16, 42);
        let queries = synthetic(20, 16, 43);
        let index = MiniHnswBuilder::new(16).m(8).ef_construction(32).build(data);

        let mut last_recall = 0.0;
        for &ef in &[1usize, 4, 16, 64, 256] {
            let mut hits = 0usize;
            let mut total = 0usize;
            for q in &queries {
                let gt: HashSet<u32> = index.brute_force(q, 10).into_iter().map(|x| x.0).collect();
                let (got, _) = index.search(q, 10, ef);
                for (id, _) in got {
                    if gt.contains(&id) {
                        hits += 1;
                    }
                }
                total += 10;
            }
            let recall = hits as f32 / total as f32;
            assert!(recall >= last_recall - 0.05, "recall must be ~monotone in ef ({recall} < {last_recall})");
            last_recall = recall;
        }
        // Big ef ⇒ near-perfect recall.
        assert!(last_recall > 0.85, "ef=256 must achieve >85% recall, got {last_recall}");
    }

    #[test]
    fn distance_evaluations_grows_with_ef() {
        let data = synthetic(500, 16, 7);
        let q = synthetic(1, 16, 99).pop().unwrap();
        let index = MiniHnswBuilder::new(16).m(8).ef_construction(32).build(data);
        let (_, s1) = index.search(&q, 10, 8);
        let (_, s2) = index.search(&q, 10, 128);
        assert!(s2.distance_evaluations > s1.distance_evaluations);
    }
}
