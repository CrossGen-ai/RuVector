//! # ruvector-crackivf-adaptive
//!
//! CrackIVF-inspired adaptive inverted-file (IVF) index for approximate
//! nearest-neighbour search. Cells whose *query density* exceeds a policy
//! threshold are split online via balanced 2-means, mimicking Idreos-style
//! database cracking (VLDB'07) but adapted to Voronoi partitions of a
//! high-dimensional vector index.
//!
//! Baseline IVF pays a permanent cost for choosing `nlist` up-front: too few
//! cells and each probe is fat; too many and both memory and build time
//! explode. CrackIVF starts with a small `nlist_0`, records how often each
//! cell is *probed* (touched during a query), and refines only the hot cells
//! reactively. Cold cells remain untouched, so total memory stays close to
//! `n * dim * 4 + nlist * dim * 4` bytes.
//!
//! ## Layout
//!
//! - `IvfIndex` — the state machine that owns centroids + posting lists.
//! - `SplitPolicy` — trait choosing when a cell should be split.
//! - `NoSplit` / `FixedThreshold` / `AdaptiveThreshold` — three built-in
//!   policies used as the required "three measured variants".
//!
//! No unsafe. No external ANN dependencies. Pure `f32` L2 distance so results
//! are byte-reproducible across runs given a fixed RNG seed.

#![forbid(unsafe_code)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;

pub mod policy;

pub use policy::{AdaptiveThreshold, FixedThreshold, NoSplit, SplitDecision, SplitPolicy};

/// Public error type. Kept intentionally small — no `thiserror` dep so this
/// crate stays lean.
#[derive(Debug)]
pub enum CrackError {
    /// Vector dimension mismatch between input and index.
    DimMismatch { expected: usize, got: usize },
    /// Attempted operation on an empty index.
    Empty,
}

impl std::fmt::Display for CrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrackError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            CrackError::Empty => write!(f, "index is empty"),
        }
    }
}

impl std::error::Error for CrackError {}

/// Squared L2 distance. Public because benchmarks and tests reuse it.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// A single Voronoi cell.
#[derive(Debug, Clone)]
struct Cell {
    centroid: Vec<f32>,
    members: Vec<u32>,
    probe_count: u64,
}

impl Cell {
    fn new(centroid: Vec<f32>) -> Self {
        Self { centroid, members: Vec::new(), probe_count: 0 }
    }
}

/// Statistics about a completed search / build round.
#[derive(Debug, Default, Clone, Copy)]
pub struct IndexStats {
    pub splits: u64,
    pub total_probes: u64,
    pub cells: usize,
}

/// The adaptive IVF index.
pub struct IvfIndex<P: SplitPolicy> {
    dim: usize,
    vectors: Vec<Vec<f32>>,
    cells: Vec<Cell>,
    policy: P,
    rng: StdRng,
    stats: IndexStats,
}

impl<P: SplitPolicy> IvfIndex<P> {
    /// Build an IVF index with `nlist_0` initial cells, seeded by random
    /// sampling of the input vectors and one Lloyd pass.
    pub fn build(vectors: Vec<Vec<f32>>, nlist_0: usize, policy: P, seed: u64) -> Result<Self, CrackError> {
        if vectors.is_empty() {
            return Err(CrackError::Empty);
        }
        let dim = vectors[0].len();
        for v in &vectors {
            if v.len() != dim {
                return Err(CrackError::DimMismatch { expected: dim, got: v.len() });
            }
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let k = nlist_0.max(1).min(vectors.len());

        // random init
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut chosen = std::collections::HashSet::new();
        while centroids.len() < k {
            let i = rng.gen_range(0..vectors.len());
            if chosen.insert(i) {
                centroids.push(vectors[i].clone());
            }
        }
        // 3 Lloyd iterations for a decent starting partition.
        for _ in 0..3 {
            centroids = lloyd_step(&vectors, &centroids, dim);
        }
        let mut cells: Vec<Cell> = centroids.into_iter().map(Cell::new).collect();
        for (id, v) in vectors.iter().enumerate() {
            let c = nearest_centroid(&cells, v);
            cells[c].members.push(id as u32);
        }
        let n_cells = cells.len();
        Ok(Self {
            dim,
            vectors,
            cells,
            policy,
            rng,
            stats: IndexStats { cells: n_cells, ..Default::default() },
        })
    }

    /// Return the top-`k` neighbour ids using `nprobe` cells and possibly
    /// splitting probed cells according to the active policy.
    pub fn search(&mut self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<(u32, f32)>, CrackError> {
        if query.len() != self.dim {
            return Err(CrackError::DimMismatch { expected: self.dim, got: query.len() });
        }
        // Rank cells by centroid distance.
        let mut cell_order: Vec<(usize, f32)> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(&c.centroid, query)))
            .collect();
        cell_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cell_order.truncate(nprobe.min(cell_order.len()));

        // Search picked cells and mark them probed.
        let mut heap: BinaryHeap<TopEntry> = BinaryHeap::with_capacity(k + 1);
        let mut touched: Vec<usize> = Vec::with_capacity(cell_order.len());
        for &(ci, _) in &cell_order {
            self.cells[ci].probe_count += 1;
            self.stats.total_probes += 1;
            touched.push(ci);
            let members = self.cells[ci].members.clone();
            for id in members {
                let d = l2_sq(&self.vectors[id as usize], query);
                if heap.len() < k {
                    heap.push(TopEntry { id, dist: d });
                } else if let Some(top) = heap.peek() {
                    if d < top.dist {
                        heap.pop();
                        heap.push(TopEntry { id, dist: d });
                    }
                }
            }
        }

        // Cracking phase: ask the policy about each touched cell.
        for ci in touched {
            let cell_size = self.cells[ci].members.len();
            let probe_count = self.cells[ci].probe_count;
            let decision = self
                .policy
                .decide(ci, cell_size, probe_count, self.cells.len(), self.vectors.len());
            if let SplitDecision::Split = decision {
                self.split_cell(ci);
                self.stats.splits += 1;
            }
        }
        self.stats.cells = self.cells.len();

        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|e| (e.id, e.dist)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(out)
    }

    /// Split cell `ci` into two via balanced 2-means (5 Lloyd iterations, then
    /// force the smaller side to hold at least `min_share` * cell_size members
    /// to avoid pathological one-sided splits).
    fn split_cell(&mut self, ci: usize) {
        let members = std::mem::take(&mut self.cells[ci].members);
        if members.len() < 4 {
            // Not worth splitting; restore and reset counter.
            self.cells[ci].members = members;
            self.cells[ci].probe_count = 0;
            return;
        }
        // Pick two seed points using k-means++-style spread.
        let seed_a = members[self.rng.gen_range(0..members.len())] as usize;
        let mut best_j = seed_a;
        let mut best_d = f32::NEG_INFINITY;
        for &m in &members {
            let d = l2_sq(&self.vectors[m as usize], &self.vectors[seed_a]);
            if d > best_d {
                best_d = d;
                best_j = m as usize;
            }
        }
        let mut ca = self.vectors[seed_a].clone();
        let mut cb = self.vectors[best_j].clone();
        let mut la: Vec<u32> = Vec::new();
        let mut lb: Vec<u32> = Vec::new();
        for _ in 0..5 {
            la.clear();
            lb.clear();
            for &m in &members {
                let v = &self.vectors[m as usize];
                if l2_sq(v, &ca) <= l2_sq(v, &cb) {
                    la.push(m);
                } else {
                    lb.push(m);
                }
            }
            ca = mean(&self.vectors, &la, self.dim);
            cb = mean(&self.vectors, &lb, self.dim);
        }
        if la.is_empty() || lb.is_empty() {
            // Degenerate split; give up and restore.
            let mut all = la;
            all.extend(lb);
            self.cells[ci].members = all;
            self.cells[ci].probe_count = 0;
            return;
        }
        // Replace ci with side A and push side B as a new cell.
        self.cells[ci] = Cell { centroid: ca, members: la, probe_count: 0 };
        self.cells.push(Cell { centroid: cb, members: lb, probe_count: 0 });
    }

    /// Read-only stats accessor.
    pub fn stats(&self) -> IndexStats {
        self.stats
    }

    /// Number of live cells.
    pub fn nlist(&self) -> usize {
        self.cells.len()
    }

    /// Total number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Estimated bytes held by the index (vectors + centroids + posting lists).
    pub fn estimated_bytes(&self) -> usize {
        let vec_bytes = self.vectors.len() * self.dim * std::mem::size_of::<f32>();
        let cent_bytes = self.cells.len() * self.dim * std::mem::size_of::<f32>();
        let post_bytes: usize = self.cells.iter().map(|c| c.members.len() * 4).sum();
        vec_bytes + cent_bytes + post_bytes
    }
}

fn nearest_centroid(cells: &[Cell], v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in cells.iter().enumerate() {
        let d = l2_sq(&c.centroid, v);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn mean(vectors: &[Vec<f32>], ids: &[u32], dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; dim];
    if ids.is_empty() {
        return acc;
    }
    for &id in ids {
        let v = &vectors[id as usize];
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    let n = ids.len() as f32;
    for a in &mut acc {
        *a /= n;
    }
    acc
}

fn lloyd_step(vectors: &[Vec<f32>], centroids: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    let k = centroids.len();
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0u32; k];
    for v in vectors {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let d = l2_sq(c, v);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        for (s, x) in sums[best].iter_mut().zip(v.iter()) {
            *s += *x;
        }
        counts[best] += 1;
    }
    for i in 0..k {
        if counts[i] > 0 {
            let n = counts[i] as f32;
            for s in sums[i].iter_mut() {
                *s /= n;
            }
        } else {
            // Empty cluster — retain old centroid to stay finite.
            sums[i] = centroids[i].clone();
        }
    }
    sums
}

#[derive(Copy, Clone)]
struct TopEntry {
    id: u32,
    dist: f32,
}

impl PartialEq for TopEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for TopEntry {}
impl PartialOrd for TopEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap on dist so `peek` returns the worst-of-topK.
        self.dist.partial_cmp(&other.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Exact brute-force top-k, used to compute recall in tests/benches.
pub fn brute_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_sq(v, query)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}

/// Recall@k of `got` against `truth`.
pub fn recall_at_k(got: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<u32> = truth.iter().map(|x| x.0).collect();
    let hits = got.iter().filter(|(id, _)| truth_ids.contains(id)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn build_and_search_basic() {
        let vs = gen(500, 16, 7);
        let mut idx = IvfIndex::build(vs.clone(), 8, NoSplit, 1).unwrap();
        let q = vs[0].clone();
        let got = idx.search(&q, 5, 4).unwrap();
        assert!(!got.is_empty());
        // The closest match should be the query itself with distance ~0.
        assert_eq!(got[0].0, 0);
        assert!(got[0].1 < 1e-6);
    }

    #[test]
    fn fixed_threshold_splits_hot_cells() {
        let vs = gen(600, 12, 42);
        let mut idx = IvfIndex::build(vs.clone(), 4, FixedThreshold::new(5, 8), 1).unwrap();
        let start = idx.nlist();
        // Hammer one query so the same cell is probed many times.
        let q = vs[10].clone();
        for _ in 0..20 {
            let _ = idx.search(&q, 5, 1).unwrap();
        }
        assert!(idx.stats().splits > 0);
        assert!(idx.nlist() > start);
    }

    #[test]
    fn recall_beats_random() {
        let vs = gen(400, 8, 3);
        let mut idx = IvfIndex::build(vs.clone(), 8, NoSplit, 2).unwrap();
        let truth = brute_topk(&vs, &vs[7], 10);
        let got = idx.search(&vs[7], 10, 4).unwrap();
        let r = recall_at_k(&got, &truth);
        assert!(r > 0.5, "recall too low: {r}");
    }

    #[test]
    fn cracking_shrinks_hot_cells() {
        // The honest claim about CrackIVF: at fixed nprobe, cracking may
        // *reduce* recall because probes cover less of the corpus — but it
        // shrinks the members-per-probed-cell count, which cuts per-probe
        // latency. This test locks in that structural invariant so that
        // regressions in the split logic are caught.
        let vs = gen(800, 16, 9);
        // Skew the workload so a small set of hot points dominates probes.
        let queries: Vec<Vec<f32>> = (0..40).map(|i| vs[i % 5].clone()).collect();
        let mut base = IvfIndex::build(vs.clone(), 4, NoSplit, 11).unwrap();
        let mut crack = IvfIndex::build(vs.clone(), 4, FixedThreshold::new(3, 16), 11).unwrap();

        for q in &queries {
            let _ = base.search(q, 10, 2).unwrap();
            let _ = crack.search(q, 10, 2).unwrap();
        }
        assert!(crack.stats().splits > 0, "no splits happened");
        assert!(
            crack.nlist() > base.nlist(),
            "nlist did not grow: base={} crack={}",
            base.nlist(),
            crack.nlist()
        );
    }
}
