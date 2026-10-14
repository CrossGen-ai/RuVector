//! IVF Ball Pruning: centroid triangle-inequality pruning for IVF probe scheduling.
//!
//! Given query q, nprobe clusters sorted by d(q, c_i). For each cluster we track r_i,
//! the max member-to-centroid distance. The best-case (minimum) distance from q to any
//! member of cluster i is `max(0, d(q,c_i) - r_i)`. If this lower bound exceeds the
//! current top-k worst score, the entire cluster can be skipped.
//!
//! Three variants are implemented behind a common `IvfScan` trait:
//!   - `BaselineScan`: scan every selected probe in full (no pruning).
//!   - `PartialAbandonScan`: per-vector partial-L2 early abandon (prior art baseline).
//!   - `BallPruneScan`: centroid-ball triangle pruning + per-vector early abandon.

use std::cmp::Ordering;

pub type VecF32 = Vec<f32>;

#[derive(Clone, Debug)]
pub struct IvfIndex {
    pub dim: usize,
    pub centroids: Vec<VecF32>,
    pub posts: Vec<Vec<u32>>, // vector ids per cluster
    pub vectors: Vec<VecF32>,
    pub r_max: Vec<f32>, // max member-to-centroid distance per cluster
}

impl IvfIndex {
    pub fn build(vectors: Vec<VecF32>, centroids: Vec<VecF32>) -> Self {
        let dim = centroids[0].len();
        let mut posts = vec![Vec::new(); centroids.len()];
        for (id, v) in vectors.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (ci, c) in centroids.iter().enumerate() {
                let d = l2_sq(v, c);
                if d < best_d {
                    best_d = d;
                    best = ci;
                }
            }
            posts[best].push(id as u32);
        }
        let mut r_max = vec![0f32; centroids.len()];
        for (ci, c) in centroids.iter().enumerate() {
            let mut m: f32 = 0.0;
            for &id in &posts[ci] {
                let d = l2_sq(&vectors[id as usize], c).sqrt();
                if d > m {
                    m = d;
                }
            }
            r_max[ci] = m;
        }
        IvfIndex { dim, centroids, posts, vectors, r_max }
    }

    pub fn mean_cluster_size(&self) -> f32 {
        let n: usize = self.posts.iter().map(|p| p.len()).sum();
        n as f32 / self.posts.len() as f32
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Partial L2 that returns early if accumulator exceeds `cutoff` squared.
#[inline]
pub fn l2_sq_abandon(a: &[f32], b: &[f32], cutoff_sq: f32) -> Option<f32> {
    let mut acc = 0f32;
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        for k in 0..8 {
            let d = a[i + k] - b[i + k];
            acc += d * d;
        }
        if acc > cutoff_sq {
            return None;
        }
        i += 8;
    }
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    if acc > cutoff_sq {
        None
    } else {
        Some(acc)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScanStats {
    pub probes_scheduled: usize,
    pub probes_actually_scanned: usize,
    pub vectors_touched: usize,
    pub full_distances: usize,
}

impl ScanStats {
    pub fn zero() -> Self {
        Self {
            probes_scheduled: 0,
            probes_actually_scanned: 0,
            vectors_touched: 0,
            full_distances: 0,
        }
    }
}

pub trait IvfScan {
    fn name(&self) -> &'static str;
    fn search(&self, idx: &IvfIndex, q: &[f32], nprobe: usize, k: usize) -> (Vec<(u32, f32)>, ScanStats);
}

pub struct BaselineScan;
pub struct PartialAbandonScan;
pub struct BallPruneScan;

fn pick_probes(idx: &IvfIndex, q: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
    let mut cd: Vec<(usize, f32)> = idx
        .centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(q, c).sqrt()))
        .collect();
    cd.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    cd.truncate(nprobe);
    cd
}

fn heap_push(heap: &mut Vec<(u32, f32)>, k: usize, id: u32, d_sq: f32) {
    if heap.len() < k {
        heap.push((id, d_sq));
        heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    } else if d_sq < heap[0].1 {
        heap[0] = (id, d_sq);
        heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    }
}

fn worst_sq(heap: &[(u32, f32)], k: usize) -> f32 {
    if heap.len() < k {
        f32::INFINITY
    } else {
        heap[0].1
    }
}

fn finalize(mut heap: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
    heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    heap
}

impl IvfScan for BaselineScan {
    fn name(&self) -> &'static str { "baseline" }
    fn search(&self, idx: &IvfIndex, q: &[f32], nprobe: usize, k: usize) -> (Vec<(u32, f32)>, ScanStats) {
        let probes = pick_probes(idx, q, nprobe);
        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut stats = ScanStats::zero();
        stats.probes_scheduled = probes.len();
        for (ci, _dc) in probes {
            stats.probes_actually_scanned += 1;
            for &id in &idx.posts[ci] {
                let d = l2_sq(q, &idx.vectors[id as usize]);
                stats.vectors_touched += 1;
                stats.full_distances += 1;
                heap_push(&mut heap, k, id, d);
            }
        }
        (finalize(heap), stats)
    }
}

impl IvfScan for PartialAbandonScan {
    fn name(&self) -> &'static str { "partial-abandon" }
    fn search(&self, idx: &IvfIndex, q: &[f32], nprobe: usize, k: usize) -> (Vec<(u32, f32)>, ScanStats) {
        let probes = pick_probes(idx, q, nprobe);
        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut stats = ScanStats::zero();
        stats.probes_scheduled = probes.len();
        for (ci, _dc) in probes {
            stats.probes_actually_scanned += 1;
            for &id in &idx.posts[ci] {
                stats.vectors_touched += 1;
                let cutoff = worst_sq(&heap, k);
                match l2_sq_abandon(q, &idx.vectors[id as usize], cutoff) {
                    Some(d) => {
                        stats.full_distances += 1;
                        heap_push(&mut heap, k, id, d);
                    }
                    None => {}
                }
            }
        }
        (finalize(heap), stats)
    }
}

impl IvfScan for BallPruneScan {
    fn name(&self) -> &'static str { "ball-prune" }
    fn search(&self, idx: &IvfIndex, q: &[f32], nprobe: usize, k: usize) -> (Vec<(u32, f32)>, ScanStats) {
        let probes = pick_probes(idx, q, nprobe);
        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut stats = ScanStats::zero();
        stats.probes_scheduled = probes.len();
        for (ci, dc) in probes {
            // Lower bound distance from q to any member of cluster ci:
            //   lb = max(0, d(q, c_ci) - r_max_ci)
            // Compare lb^2 to current top-k worst (squared).
            let lb = (dc - idx.r_max[ci]).max(0.0);
            let lb_sq = lb * lb;
            if lb_sq >= worst_sq(&heap, k) {
                continue; // prune entire cluster
            }
            stats.probes_actually_scanned += 1;
            for &id in &idx.posts[ci] {
                stats.vectors_touched += 1;
                let cutoff = worst_sq(&heap, k);
                match l2_sq_abandon(q, &idx.vectors[id as usize], cutoff) {
                    Some(d) => {
                        stats.full_distances += 1;
                        heap_push(&mut heap, k, id, d);
                    }
                    None => {}
                }
            }
        }
        (finalize(heap), stats)
    }
}

/// Brute-force exact top-k for recall measurement.
pub fn exact_topk(vectors: &[VecF32], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (id, v) in vectors.iter().enumerate() {
        let d = l2_sq(q, v);
        heap_push(&mut heap, k, id as u32, d);
    }
    finalize(heap)
}

pub fn recall(found: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().map(|(i, _)| *i).collect();
    let hits = found.iter().filter(|(i, _)| t.contains(i)).count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<VecF32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    fn kmeans(points: &[VecF32], k: usize, iters: usize, seed: u64) -> Vec<VecF32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = points[0].len();
        let mut centroids: Vec<VecF32> = (0..k)
            .map(|_| points[rng.gen_range(0..points.len())].clone())
            .collect();
        for _ in 0..iters {
            let mut sums = vec![vec![0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for p in points {
                let mut best = 0;
                let mut bd = f32::INFINITY;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = l2_sq(p, c);
                    if d < bd { bd = d; best = ci; }
                }
                counts[best] += 1;
                for d in 0..dim { sums[best][d] += p[d]; }
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    for d in 0..dim { centroids[ci][d] = sums[ci][d] / counts[ci] as f32; }
                }
            }
        }
        centroids
    }

    #[test]
    fn ball_prune_matches_baseline() {
        let data = gen_data(2000, 32, 1);
        let queries = gen_data(20, 32, 2);
        let centroids = kmeans(&data, 32, 10, 3);
        let idx = IvfIndex::build(data.clone(), centroids);

        for q in &queries {
            let (base, _) = BaselineScan.search(&idx, q, 32, 10);
            let (ball, _) = BallPruneScan.search(&idx, q, 32, 10);
            // With nprobe = all clusters, ball prune should give identical top-k.
            let bset: std::collections::HashSet<u32> = base.iter().map(|(i, _)| *i).collect();
            let gset: std::collections::HashSet<u32> = ball.iter().map(|(i, _)| *i).collect();
            assert_eq!(bset, gset, "ball prune lost results at nprobe=all");
        }
    }

    #[test]
    fn ball_prune_reduces_touches() {
        let data = gen_data(4000, 32, 11);
        let queries = gen_data(50, 32, 12);
        let centroids = kmeans(&data, 64, 10, 13);
        let idx = IvfIndex::build(data.clone(), centroids);

        let mut baseline_touches = 0usize;
        let mut ball_touches = 0usize;
        for q in &queries {
            let (_, s1) = BaselineScan.search(&idx, q, 64, 10);
            let (_, s2) = BallPruneScan.search(&idx, q, 64, 10);
            baseline_touches += s1.vectors_touched;
            ball_touches += s2.vectors_touched;
        }
        assert!(ball_touches < baseline_touches,
            "ball prune should touch fewer vectors ({} vs {})",
            ball_touches, baseline_touches);
    }

    #[test]
    fn partial_abandon_matches_baseline_topk() {
        let data = gen_data(1500, 24, 21);
        let queries = gen_data(10, 24, 22);
        let centroids = kmeans(&data, 24, 8, 23);
        let idx = IvfIndex::build(data.clone(), centroids);
        for q in &queries {
            let (b, _) = BaselineScan.search(&idx, q, 24, 5);
            let (p, _) = PartialAbandonScan.search(&idx, q, 24, 5);
            let bset: std::collections::HashSet<u32> = b.iter().map(|(i, _)| *i).collect();
            let pset: std::collections::HashSet<u32> = p.iter().map(|(i, _)| *i).collect();
            assert_eq!(bset, pset);
        }
    }
}
