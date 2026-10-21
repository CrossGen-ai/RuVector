//! IVF probe-loop early termination via per-cell radius + triangle lower bound.
//!
//! Standard IVF scans a user-chosen `nprobe` cells for every query. For many
//! queries, the top-k is settled well before all `nprobe` cells are visited:
//! the remaining cells' centroids are already far enough from the query that
//! no point in them could beat the current k-th-best candidate.
//!
//! We make that statement provable with the triangle inequality. If cell `c`
//! has centroid `mu_c` and radius `r_c = max_{p in c} ||p - mu_c||`, then for
//! any query `q` and any point `p` in cell `c`:
//!
//!   ||q - p|| >= ||q - mu_c|| - r_c     (Euclidean triangle inequality)
//!
//! So the moment we observe `||q - mu_next|| - r_next >= tau_k`, where
//! `tau_k` is the current k-th-best candidate distance, every yet-unvisited
//! cell is also safe to skip (we scan cells in increasing `||q - mu_c||`).
//! That cut is **exact w.r.t. the IVF partition**: it does not change which
//! points IVF would have returned if it had kept scanning.
//!
//! This crate ships three `IvfSearcher` backends over the same `IvfIndex`:
//!   * `FixedNprobe`   — baseline: scan the top-`nprobe` cells fully.
//!   * `RadiusET`      — scan cells in order; stop when LB_c >= tau_k.
//!   * `HybridNprobeET`— combine a `nprobe` cap with the ET cut.
//!
//! All backends are deterministic given the same index and query.

#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*). Zero external deps so the crate is light.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        // uniform [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Box-Muller unit normal.
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-9).min(1.0 - 1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Distance utilities
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline(always)]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    sq_l2(a, b).sqrt()
}

// ---------------------------------------------------------------------------
// Flat vector store
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct FlatVectors {
    pub dim: usize,
    pub data: Vec<f32>, // row-major, n*dim
}

impl FlatVectors {
    pub fn new(dim: usize) -> Self {
        Self { dim, data: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

// ---------------------------------------------------------------------------
// IVF index
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct IvfIndex {
    pub dim: usize,
    pub nlist: usize,
    /// Row-major centroids (nlist * dim).
    pub centroids: Vec<f32>,
    /// posting_lists[c] = global ids of points assigned to cell c.
    pub posting_lists: Vec<Vec<u32>>,
    /// Per-cell max radius (Euclidean) from centroid to any member.
    pub cell_radii: Vec<f32>,
    /// Flat vector store (owned copy of input points).
    pub vectors: FlatVectors,
}

impl IvfIndex {
    /// Build an IVF index with simple Lloyd k-means on `vectors`.
    ///
    /// Returns a trained + populated index. `iters` is the number of Lloyd
    /// passes; 10 is usually enough for benchmarking.
    pub fn train(
        vectors: FlatVectors,
        nlist: usize,
        iters: usize,
        seed: u64,
    ) -> Self {
        let dim = vectors.dim;
        let n = vectors.len();
        assert!(nlist >= 1 && nlist <= n, "nlist out of range");

        let mut rng = Rng::new(seed);
        let mut centroids = vec![0.0_f32; nlist * dim];

        // Random seeding: pick `nlist` distinct input points.
        let mut taken = vec![false; n];
        for c in 0..nlist {
            let mut idx;
            loop {
                idx = (rng.next_u64() as usize) % n;
                if !taken[idx] {
                    taken[idx] = true;
                    break;
                }
            }
            centroids[c * dim..(c + 1) * dim].copy_from_slice(vectors.row(idx));
        }

        // Lloyd iterations.
        let mut assign = vec![0u32; n];
        for _ in 0..iters {
            // Assign.
            for i in 0..n {
                let v = vectors.row(i);
                let (mut best, mut best_d) = (0u32, f32::INFINITY);
                for c in 0..nlist {
                    let cen = &centroids[c * dim..(c + 1) * dim];
                    let d = sq_l2(v, cen);
                    if d < best_d {
                        best_d = d;
                        best = c as u32;
                    }
                }
                assign[i] = best;
            }
            // Update.
            let mut counts = vec![0u32; nlist];
            let mut sums = vec![0.0_f32; nlist * dim];
            for i in 0..n {
                let c = assign[i] as usize;
                counts[c] += 1;
                let base = c * dim;
                let v = vectors.row(i);
                for d in 0..dim {
                    sums[base + d] += v[d];
                }
            }
            for c in 0..nlist {
                if counts[c] == 0 {
                    continue; // keep previous centroid for empty cells
                }
                let inv = 1.0 / counts[c] as f32;
                let base = c * dim;
                for d in 0..dim {
                    centroids[base + d] = sums[base + d] * inv;
                }
            }
        }

        // Final posting lists + per-cell radii.
        let mut posting_lists = vec![Vec::<u32>::new(); nlist];
        let mut cell_radii = vec![0.0_f32; nlist];
        for i in 0..n {
            let v = vectors.row(i);
            let (mut best, mut best_sq) = (0usize, f32::INFINITY);
            for c in 0..nlist {
                let cen = &centroids[c * dim..(c + 1) * dim];
                let d = sq_l2(v, cen);
                if d < best_sq {
                    best_sq = d;
                    best = c;
                }
            }
            posting_lists[best].push(i as u32);
            let r = best_sq.sqrt();
            if r > cell_radii[best] {
                cell_radii[best] = r;
            }
        }

        Self {
            dim,
            nlist,
            centroids,
            posting_lists,
            cell_radii,
            vectors,
        }
    }

    #[inline]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }
}

// ---------------------------------------------------------------------------
// Top-K heap (max-heap keyed by distance; drops the largest-distance element).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool { self.dist == o.dist }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { self.dist.partial_cmp(&o.dist) }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}

pub struct TopK {
    k: usize,
    heap: std::collections::BinaryHeap<Candidate>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, heap: std::collections::BinaryHeap::with_capacity(k + 1) }
    }
    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push(Candidate { dist, id });
        } else if let Some(top) = self.heap.peek() {
            if dist < top.dist {
                self.heap.pop();
                self.heap.push(Candidate { dist, id });
            }
        }
    }
    /// Current k-th-best distance (tau_k). Returns +inf if fewer than k seen.
    #[inline]
    pub fn tau(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map(|c| c.dist).unwrap_or(f32::INFINITY)
        }
    }
    pub fn into_sorted(self) -> Vec<(f32, u32)> {
        let mut v: Vec<(f32, u32)> =
            self.heap.into_iter().map(|c| (c.dist, c.id)).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        v
    }
}

// ---------------------------------------------------------------------------
// Searcher trait + backends
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct SearchStats {
    pub cells_visited: usize,
    pub points_scanned: usize,
}

pub trait IvfSearcher {
    fn search(
        &self,
        index: &IvfIndex,
        q: &[f32],
        k: usize,
    ) -> (Vec<(f32, u32)>, SearchStats);
}

/// Baseline: scan the top-`nprobe` cells by centroid distance, with no cut.
///
/// This matches the standard IVF textbook: order cells by `||q - mu_c||`
/// ascending and visit the first `nprobe`.
pub struct FixedNprobe {
    pub nprobe: usize,
}

impl IvfSearcher for FixedNprobe {
    fn search(
        &self,
        index: &IvfIndex,
        q: &[f32],
        k: usize,
    ) -> (Vec<(f32, u32)>, SearchStats) {
        let sorted = sort_cells_by_centroid(index, q);
        let nprobe = self.nprobe.min(sorted.len());
        let mut topk = TopK::new(k);
        let mut stats = SearchStats::default();
        for item in sorted.iter().take(nprobe) {
            scan_cell(index, item.0 as usize, q, &mut topk, &mut stats);
        }
        (topk.into_sorted(), stats)
    }
}

/// Pure early-termination: scan cells in **LB order** (ascending), stop as
/// soon as the next cell's LB is >= `tau_k`.
///
/// Soundness: with cells sorted by `LB_c = max(0, ||q - mu_c|| - r_c)`
/// ascending, if cell `c` has `LB_c >= tau_k`, every remaining cell `c'`
/// has `LB_{c'} >= LB_c >= tau_k`, so no unvisited point can beat the
/// current k-th-best. Terminating is exact w.r.t. the IVF partition.
pub struct RadiusET;

impl IvfSearcher for RadiusET {
    fn search(
        &self,
        index: &IvfIndex,
        q: &[f32],
        k: usize,
    ) -> (Vec<(f32, u32)>, SearchStats) {
        let sorted = sort_cells_by_lb(index, q);
        let mut topk = TopK::new(k);
        let mut stats = SearchStats::default();
        for &(c, lb_lin) in sorted.iter() {
            // topk.tau() is sq-L2; compare on squared units.
            let lb_sq = lb_lin * lb_lin;
            if lb_sq >= topk.tau() {
                break;
            }
            scan_cell(index, c as usize, q, &mut topk, &mut stats);
        }
        (topk.into_sorted(), stats)
    }
}

/// Capped early-termination: at most `max_nprobe` cells, LB-sorted, with the
/// LB cut applied inside the cap.
pub struct HybridNprobeET {
    pub max_nprobe: usize,
}

impl IvfSearcher for HybridNprobeET {
    fn search(
        &self,
        index: &IvfIndex,
        q: &[f32],
        k: usize,
    ) -> (Vec<(f32, u32)>, SearchStats) {
        let sorted = sort_cells_by_lb(index, q);
        let cap = self.max_nprobe.min(sorted.len());
        let mut topk = TopK::new(k);
        let mut stats = SearchStats::default();
        for &(c, lb_lin) in sorted.iter().take(cap) {
            let lb_sq = lb_lin * lb_lin;
            if lb_sq >= topk.tau() {
                break;
            }
            scan_cell(index, c as usize, q, &mut topk, &mut stats);
        }
        (topk.into_sorted(), stats)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns (cell_id, linear_distance_from_q_to_centroid), sorted ascending.
/// Standard IVF visit order, used by the fixed-nprobe baseline.
fn sort_cells_by_centroid(index: &IvfIndex, q: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = (0..index.nlist)
        .map(|c| (c as u32, l2(q, index.centroid(c))))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    v
}

/// Returns (cell_id, LB = max(0, ||q - mu_c|| - r_c)), sorted ascending.
///
/// This is the correct visit order for the LB-based early-termination cut:
/// if we scan cells in increasing LB and terminate at LB_c >= tau_k, every
/// unvisited cell c' has LB_{c'} >= LB_c >= tau_k, so no unvisited point
/// can beat tau_k.
fn sort_cells_by_lb(index: &IvfIndex, q: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = (0..index.nlist)
        .map(|c| {
            let dc = l2(q, index.centroid(c));
            let lb = (dc - index.cell_radii[c]).max(0.0);
            (c as u32, lb)
        })
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    v
}

#[inline]
fn scan_cell(
    index: &IvfIndex,
    c: usize,
    q: &[f32],
    topk: &mut TopK,
    stats: &mut SearchStats,
) {
    let plist = &index.posting_lists[c];
    stats.cells_visited += 1;
    stats.points_scanned += plist.len();
    for &pid in plist {
        let p = index.vectors.row(pid as usize);
        let d = sq_l2(q, p);
        topk.push(d, pid);
    }
}

// ---------------------------------------------------------------------------
// Ground-truth / recall helpers (used in tests + benches)
// ---------------------------------------------------------------------------

pub fn brute_force_topk(vectors: &FlatVectors, q: &[f32], k: usize) -> Vec<(f32, u32)> {
    let mut topk = TopK::new(k);
    for i in 0..vectors.len() {
        let d = sq_l2(q, vectors.row(i));
        topk.push(d, i as u32);
    }
    topk.into_sorted()
}

pub fn recall_at_k(gt: &[(f32, u32)], got: &[(f32, u32)]) -> f32 {
    if gt.is_empty() {
        return 1.0;
    }
    let want: std::collections::HashSet<u32> = gt.iter().map(|x| x.1).collect();
    let mut hit = 0usize;
    for (_, id) in got {
        if want.contains(id) {
            hit += 1;
        }
    }
    hit as f32 / gt.len() as f32
}

// Tests live in `tests/correctness.rs` to keep lib.rs under 500 lines.
