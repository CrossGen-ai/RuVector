//! RoarGraph-inspired out-of-distribution (OOD) ANN.
//!
//! Cross-modal RAG (e.g. text-query → image-embedding base) suffers because
//! the query distribution diverges from the base distribution: graph indexes
//! built purely over base vectors route queries through low-similarity
//! regions, hurting recall at fixed compute budgets.
//!
//! Chen et al. ("RoarGraph: A Projected Bipartite Graph for Efficient
//! Cross-Modal ANNS", VLDB 2024) attack this by (1) sampling a *query
//! workload* and (2) projecting each query's `k` nearest base vectors into a
//! bipartite graph, then folding that structure into the base-only graph so
//! greedy walks starting from base points can reach the OOD-relevant
//! neighbourhoods faster.
//!
//! This crate implements a compact, faithful-in-spirit variant we call
//! **RoarGraph-lite**, comparing three swappable trait-based backends:
//!
//! 1. [`FlatIndex`]     — exact brute-force baseline (O(n·d) per query).
//! 2. [`KnnGraphIndex`] — greedy walk over a base-only k-NN graph.
//! 3. [`RoarGraphIndex`] — base-only k-NN graph augmented with edges
//!    derived from a projected query-workload bipartite graph.
//!
//! All three implement the [`AnnIndex`] trait so downstream code can swap
//! backends. No external crates, no unsafe, no mocks.

use std::collections::{BinaryHeap, HashSet};

// ─── Basic types ─────────────────────────────────────────────────────────────

pub type Vector = Vec<f32>;

/// One indexed entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u32,
    pub vec: Vector,
}

/// A search hit (id + similarity score).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

// ─── Distance ────────────────────────────────────────────────────────────────

/// Cosine similarity in [-1, 1]. Higher is better.
///
/// Both inputs must have the same non-zero length and non-zero norm; violating
/// either is a programmer error and returns 0.0 defensively.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Swappable ANN backend contract.
pub trait AnnIndex {
    /// Return top-k hits sorted by descending score.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    /// Estimated resident bytes (excluding vector payload duplication).
    fn memory_bytes(&self) -> usize;
    /// Human-readable name for reporting.
    fn name(&self) -> &'static str;
}

// ─── Reproducible RNG (xorshift64*) ──────────────────────────────────────────

/// Tiny deterministic PRNG — avoids pulling `rand` for zero-dep policy.
pub struct XorShift {
    state: u64,
}

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdead_beef_cafe_babe } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        // Uniform in [0, 1)
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
    /// Approximate standard-normal via Box–Muller.
    pub fn next_gauss(&mut self) -> f32 {
        loop {
            let u1 = self.next_f32().max(1e-9);
            let u2 = self.next_f32();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            if z.is_finite() {
                return z;
            }
        }
    }
}

// ─── Dataset generation (base + OOD queries + ground truth) ──────────────────

/// Generate a synthetic cross-modal-style dataset.
///
/// Base vectors are drawn from N(0, 1) then shifted by `+base_shift` on each
/// dimension; queries are drawn from N(0, 1) shifted by `-query_shift`.
/// The signed asymmetry simulates the OOD gap between a text query encoder
/// and an image base encoder: the two clouds have overlapping support but
/// distinct centroids, and greedy walks over base-only graphs pay for it.
pub fn gen_dataset(
    n_base: usize,
    n_query: usize,
    dim: usize,
    base_shift: f32,
    query_shift: f32,
    seed: u64,
) -> (Vec<Entry>, Vec<Entry>) {
    let mut rng = XorShift::new(seed);
    let mut base = Vec::with_capacity(n_base);
    for id in 0..n_base {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(rng.next_gauss() + base_shift);
        }
        base.push(Entry { id: id as u32, vec: v });
    }
    let mut queries = Vec::with_capacity(n_query);
    for id in 0..n_query {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(rng.next_gauss() - query_shift);
        }
        queries.push(Entry { id: id as u32, vec: v });
    }
    (base, queries)
}

/// Compute exact top-k ground truth for a query set (used for recall).
pub fn ground_truth(base: &[Entry], queries: &[Entry], k: usize) -> Vec<Vec<Hit>> {
    let flat = FlatIndex::build(base);
    queries.iter().map(|q| flat.search(&q.vec, k)).collect()
}

/// Recall@k of `got` vs `gold`, treated as sets of ids.
pub fn recall_at_k(got: &[Hit], gold: &[Hit]) -> f32 {
    if gold.is_empty() {
        return 1.0;
    }
    let gold_ids: HashSet<u32> = gold.iter().map(|h| h.id).collect();
    let hit = got.iter().filter(|h| gold_ids.contains(&h.id)).count();
    hit as f32 / gold_ids.len() as f32
}

// ─── Heap helpers ────────────────────────────────────────────────────────────

/// Max-heap entry keyed by score (for top-k retention).
#[derive(Copy, Clone)]
struct HeapItem(f32, u32);

impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // NaN-safe partial → total via total_cmp on the score.
        self.0.total_cmp(&o.0).then_with(|| self.1.cmp(&o.1))
    }
}

fn topk_sorted(mut heap: BinaryHeap<HeapItem>, k: usize) -> Vec<Hit> {
    let mut v: Vec<Hit> = Vec::with_capacity(k);
    while let Some(HeapItem(s, id)) = heap.pop() {
        v.push(Hit { id, score: s });
        if v.len() >= k {
            break;
        }
    }
    v
}

// ─── Flat (exact) baseline ───────────────────────────────────────────────────

pub struct FlatIndex {
    entries: Vec<Entry>,
}

impl FlatIndex {
    pub fn build(entries: &[Entry]) -> Self {
        Self { entries: entries.to_vec() }
    }
}

impl AnnIndex for FlatIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(self.entries.len());
        for e in &self.entries {
            heap.push(HeapItem(cosine(query, &e.vec), e.id));
        }
        topk_sorted(heap, k)
    }
    fn memory_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<Entry>()
            + self
                .entries
                .iter()
                .map(|e| e.vec.len() * 4)
                .sum::<usize>()
    }
    fn name(&self) -> &'static str {
        "FlatIndex"
    }
}

// ─── k-NN graph baseline ─────────────────────────────────────────────────────

/// Base-only greedy graph search over a static k-NN graph.
///
/// This is intentionally simpler than HNSW/NSG — it isolates the effect of
/// query-workload-aware augmentation (the variable RoarGraph adds).
pub struct KnnGraphIndex {
    entries: Vec<Entry>,
    /// Adjacency: `graph[i]` = out-neighbours of node `i`.
    graph: Vec<Vec<u32>>,
    /// Entry points for greedy walk (random sample; deterministic).
    entry_points: Vec<u32>,
    ef: usize,
}

impl KnnGraphIndex {
    /// Build a k-NN graph by exact top-k over base (O(n²·d) — acceptable at
    /// PoC scale; the search variance we care about is not build time).
    pub fn build(entries: &[Entry], k_graph: usize, ef: usize, n_entry: usize, seed: u64) -> Self {
        let flat = FlatIndex::build(entries);
        let mut graph = Vec::with_capacity(entries.len());
        for e in entries {
            // top (k+1) then drop self
            let neigh = flat.search(&e.vec, k_graph + 1);
            let mut ids: Vec<u32> = neigh
                .into_iter()
                .filter(|h| h.id != e.id)
                .take(k_graph)
                .map(|h| h.id)
                .collect();
            ids.sort_unstable();
            ids.dedup();
            graph.push(ids);
        }
        let mut rng = XorShift::new(seed);
        let mut entry_points: Vec<u32> = Vec::with_capacity(n_entry);
        for _ in 0..n_entry {
            entry_points.push((rng.next_u64() % entries.len() as u64) as u32);
        }
        Self { entries: entries.to_vec(), graph, entry_points, ef }
    }

    /// Greedy best-first walk (identical routine used by both graph indexes).
    fn greedy_search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(self.ef * 4);
        // Frontier: min-heap by (-score) so we always expand the highest sim.
        // BinaryHeap is max; store score as-is and pop max.
        let mut frontier: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(self.ef * 2);
        // Result set: bounded max-heap of size ef; we keep best-scored.
        let mut results: BinaryHeap<std::cmp::Reverse<HeapItem>> =
            BinaryHeap::with_capacity(self.ef + 1);

        for &ep in &self.entry_points {
            if visited.insert(ep) {
                let s = cosine(query, &self.entries[ep as usize].vec);
                frontier.push(HeapItem(s, ep));
                results.push(std::cmp::Reverse(HeapItem(s, ep)));
                if results.len() > self.ef {
                    results.pop();
                }
            }
        }

        while let Some(HeapItem(cur_score, cur_id)) = frontier.pop() {
            // Termination: if current best-in-frontier is worse than the
            // worst-in-results and we have ef results, stop.
            if results.len() >= self.ef {
                if let Some(std::cmp::Reverse(HeapItem(worst, _))) = results.peek() {
                    if cur_score < *worst {
                        break;
                    }
                }
            }
            for &n in &self.graph[cur_id as usize] {
                if visited.insert(n) {
                    let s = cosine(query, &self.entries[n as usize].vec);
                    frontier.push(HeapItem(s, n));
                    results.push(std::cmp::Reverse(HeapItem(s, n)));
                    if results.len() > self.ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Hit> = results
            .into_iter()
            .map(|std::cmp::Reverse(HeapItem(s, id))| Hit { id, score: s })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(k);
        out
    }
}

impl AnnIndex for KnnGraphIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.greedy_search(query, k)
    }
    fn memory_bytes(&self) -> usize {
        let vec_bytes: usize = self.entries.iter().map(|e| e.vec.len() * 4).sum();
        let graph_bytes: usize = self.graph.iter().map(|n| n.len() * 4).sum();
        vec_bytes + graph_bytes + self.entry_points.len() * 4
    }
    fn name(&self) -> &'static str {
        "KnnGraphIndex"
    }
}

// ─── RoarGraph-lite ──────────────────────────────────────────────────────────

/// RoarGraph-lite: k-NN graph augmented with query-projected bipartite edges.
///
/// Build:
///   1. Build a base-only k-NN graph (same routine as [`KnnGraphIndex`]).
///   2. Take a *query workload sample* (representative queries the operator
///      already has — in RAG this is real user traffic).
///   3. For each sample query, find its top-`k_bipartite` neighbours in the
///      base. Every unordered pair of those neighbours becomes an edge
///      candidate (query-projected bipartite → base-side co-occurrence).
///   4. Fold co-occurrence edges into each node's adjacency, capped at
///      `k_graph + k_aug` per node. Node degree stays bounded so latency
///      overhead is bounded too.
///   5. Move the entry points to base nodes that appear most often in the
///      top-1 of the workload sample — they are the natural "landing zones"
///      for OOD queries and jump-start greedy walks.
pub struct RoarGraphIndex {
    inner: KnnGraphIndex,
    /// Extra bookkeeping for honest memory accounting.
    aug_edges_added: usize,
}

impl RoarGraphIndex {
    pub fn build(
        entries: &[Entry],
        workload: &[Entry],
        k_graph: usize,
        k_bipartite: usize,
        k_aug: usize,
        ef: usize,
        n_entry: usize,
    ) -> Self {
        // Start from the base-only graph.
        let base = KnnGraphIndex::build(entries, k_graph, ef, n_entry, 0xC0FFEE);
        let flat = FlatIndex::build(entries);

        // Compute per-query top-k_bipartite (used both for augmentation and
        // entry-point selection).
        let workload_hits: Vec<Vec<Hit>> = workload
            .iter()
            .map(|q| flat.search(&q.vec, k_bipartite.max(1)))
            .collect();

        // 1) Augmentation: co-occurring pairs get an extra edge each.
        let mut graph = base.graph;
        let cap = k_graph + k_aug;
        let mut added = 0usize;
        // Track sets for dedup during insertion.
        let mut sets: Vec<HashSet<u32>> = graph
            .iter()
            .map(|v| v.iter().copied().collect())
            .collect();
        for hits in &workload_hits {
            for i in 0..hits.len() {
                for j in (i + 1)..hits.len() {
                    let a = hits[i].id;
                    let b = hits[j].id;
                    if graph[a as usize].len() < cap && sets[a as usize].insert(b) {
                        graph[a as usize].push(b);
                        added += 1;
                    }
                    if graph[b as usize].len() < cap && sets[b as usize].insert(a) {
                        graph[b as usize].push(a);
                        added += 1;
                    }
                }
            }
        }

        // 2) Better entry points: base nodes that are top-1 for many workload
        // queries. Fall back to random if workload is empty.
        let entry_points: Vec<u32> = if workload_hits.is_empty() {
            base.entry_points
        } else {
            let mut counts: std::collections::HashMap<u32, u32> = Default::default();
            for hits in &workload_hits {
                if let Some(h) = hits.first() {
                    *counts.entry(h.id).or_insert(0) += 1;
                }
            }
            let mut ranked: Vec<(u32, u32)> = counts.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));
            let mut ep: Vec<u32> = ranked.into_iter().take(n_entry).map(|(id, _)| id).collect();
            // Pad with base entry points if too few.
            for e in &base.entry_points {
                if ep.len() >= n_entry {
                    break;
                }
                if !ep.contains(e) {
                    ep.push(*e);
                }
            }
            if ep.is_empty() {
                base.entry_points
            } else {
                ep
            }
        };

        Self {
            inner: KnnGraphIndex {
                entries: base.entries,
                graph,
                entry_points,
                ef,
            },
            aug_edges_added: added,
        }
    }
    pub fn aug_edges(&self) -> usize {
        self.aug_edges_added
    }
}

impl AnnIndex for RoarGraphIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.inner.greedy_search(query, k)
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    fn name(&self) -> &'static str {
        "RoarGraphIndex"
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Vec<Entry>, Vec<Entry>) {
        gen_dataset(500, 50, 16, 0.4, 0.4, 42)
    }

    #[test]
    fn cosine_bounds() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        let c = vec![-1.0, 0.0, 0.0];
        assert!((cosine(&a, &c) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn flat_matches_ground_truth() {
        let (base, queries) = tiny();
        let idx = FlatIndex::build(&base);
        let gt = ground_truth(&base, &queries, 10);
        for (q, g) in queries.iter().zip(gt.iter()) {
            let r = idx.search(&q.vec, 10);
            // Flat is the ground truth by construction.
            assert!((recall_at_k(&r, g) - 1.0).abs() < 1e-6);
            assert_eq!(r.len(), 10);
        }
    }

    #[test]
    fn knn_graph_beats_random() {
        let (base, queries) = tiny();
        let idx = KnnGraphIndex::build(&base, 12, 32, 8, 7);
        let gt = ground_truth(&base, &queries, 10);
        let mut acc = 0.0;
        for (q, g) in queries.iter().zip(gt.iter()) {
            acc += recall_at_k(&idx.search(&q.vec, 10), g);
        }
        let mean = acc / queries.len() as f32;
        // Random top-10 out of 500 = 0.02 recall; graph must clear 0.3.
        assert!(mean > 0.3, "k-NN graph recall too low: {mean}");
    }

    #[test]
    fn roargraph_matches_or_beats_knn_graph_on_ood() {
        // OOD by construction (base +0.6, query -0.6).
        let (base, queries) = gen_dataset(600, 80, 16, 0.6, 0.6, 11);
        // Use half the queries as workload, other half for evaluation.
        let (workload, eval): (Vec<_>, Vec<_>) = queries
            .into_iter()
            .enumerate()
            .partition(|(i, _)| i % 2 == 0);
        let workload: Vec<Entry> = workload.into_iter().map(|(_, e)| e).collect();
        let eval: Vec<Entry> = eval.into_iter().map(|(_, e)| e).collect();

        let knn = KnnGraphIndex::build(&base, 12, 32, 8, 7);
        let roar = RoarGraphIndex::build(&base, &workload, 12, 6, 6, 32, 8);
        let gt = ground_truth(&base, &eval, 10);

        let mean = |idx: &dyn AnnIndex| -> f32 {
            let mut acc = 0.0;
            for (q, g) in eval.iter().zip(gt.iter()) {
                acc += recall_at_k(&idx.search(&q.vec, 10), g);
            }
            acc / eval.len() as f32
        };
        let r_knn = mean(&knn);
        let r_roar = mean(&roar);
        // Structural guarantee — RoarGraph's superset adjacency + workload-
        // aware entry points means it cannot lose to plain k-NN by more than
        // sampling noise. Allow a small epsilon.
        assert!(
            r_roar + 0.02 >= r_knn,
            "RoarGraph regressed vs k-NN: roar={r_roar}, knn={r_knn}"
        );
        assert!(roar.aug_edges() > 0);
    }
}
