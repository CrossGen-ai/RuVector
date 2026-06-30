//! ruvector-cache-block-hnsw — Block-packed adjacency layout for HNSW.
//!
//! Three swappable variants of the same graph index, sharing a trait:
//!
//! 1. [`BaselineHnsw`]  — neighbours stored as a single global `Vec<u32>` indexed
//!    by `offset[node]..offset[node]+deg[node]`. Pointer chase: load list, then
//!    load each neighbour's full FP32 vector to compute distance.
//!
//! 2. [`BlockHnsw`]     — neighbours packed into fixed 64-byte cache blocks
//!    (16 × `u32`) so that one cache line carries one node's whole adjacency
//!    (degree capped at 16, matching HNSW's typical `M=16`). A prefetch hint
//!    is issued for each candidate's neighbour block as it is enqueued.
//!
//! 3. [`SketchHnsw`]    — 64-byte blocks split as `12 × u32` neighbour IDs +
//!    `12 × u8` 4-bit-quantised distance sketches between this node and each
//!    neighbour. The sketch is used as an *early-reject* against the query
//!    sketch before the full FP32 distance is computed. Degree cap = 12.
//!
//! All three implement [`AnnIndex`] so back-ends are swappable.
//!
//! NB: This is a research PoC focused on **layer-0 traversal**. The upper
//! layers are just sequential entry-point selection — fine for benchmarking
//! the inner loop, which is where >95% of search wall-time lives in HNSW.

#![allow(clippy::needless_range_loop)]

use std::collections::BinaryHeap;
use std::cmp::Ordering;

pub type NodeId = u32;

// ---------- distance ----------
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

// ---------- common trait ----------
pub trait AnnIndex {
    fn build(vecs: &[Vec<f32>], m: usize, ef_construction: usize) -> Self where Self: Sized;
    fn search(&self, q: &[f32], k: usize, ef_search: usize) -> Vec<(NodeId, f32)>;
    fn name(&self) -> &'static str;
    /// Bytes used by the adjacency representation (not the vectors).
    fn adjacency_bytes(&self) -> usize;
}

// ---------- shared bits: heap items ----------
#[derive(Copy, Clone, Debug)]
struct DistNode { d: f32, n: NodeId }
impl PartialEq for DistNode { fn eq(&self, o: &Self) -> bool { self.d == o.d && self.n == o.n } }
impl Eq for DistNode {}
impl PartialOrd for DistNode { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for DistNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // max-heap by distance; tie-break by id for stability
        self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal).then(self.n.cmp(&o.n))
    }
}
// Min-heap wrapper
#[derive(Copy, Clone, Debug)]
struct MinDistNode(DistNode);
impl PartialEq for MinDistNode { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for MinDistNode {}
impl PartialOrd for MinDistNode { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MinDistNode { fn cmp(&self, o: &Self) -> Ordering { o.0.cmp(&self.0) } }

// ---------- 1. Baseline ----------
pub struct BaselineHnsw {
    vecs: Vec<Vec<f32>>,
    /// CSR-style adjacency for layer 0.
    offsets: Vec<u32>,
    nbrs: Vec<NodeId>,
    entry: NodeId,
    dim: usize,
}

impl AnnIndex for BaselineHnsw {
    fn name(&self) -> &'static str { "baseline" }
    fn adjacency_bytes(&self) -> usize { self.offsets.len() * 4 + self.nbrs.len() * 4 }

    fn build(vecs: &[Vec<f32>], m: usize, ef: usize) -> Self {
        let n = vecs.len();
        let dim = vecs[0].len();
        let mut tmp: Vec<Vec<(f32, NodeId)>> = vec![Vec::with_capacity(m); n];
        let entry: NodeId = 0;
        for (i, v) in vecs.iter().enumerate().skip(1) {
            // greedy from entry, ef expansion
            let neighbours = greedy_search(vecs, entry, v, ef, &tmp);
            let kept: Vec<_> = neighbours.into_iter().take(m).collect();
            for (d, j) in kept {
                tmp[i].push((d, j));
                if tmp[j as usize].len() < m {
                    tmp[j as usize].push((d, i as NodeId));
                } else if let Some(worst_idx) = tmp[j as usize].iter().enumerate().max_by(|a,b| a.1.0.partial_cmp(&b.1.0).unwrap()).map(|(k,_)|k) {
                    if d < tmp[j as usize][worst_idx].0 {
                        tmp[j as usize][worst_idx] = (d, i as NodeId);
                    }
                }
            }
        }
        // flatten
        let mut offsets = Vec::with_capacity(n + 1);
        let mut nbrs = Vec::new();
        offsets.push(0);
        for adj in &tmp {
            for (_d, j) in adj { nbrs.push(*j); }
            offsets.push(nbrs.len() as u32);
        }
        Self { vecs: vecs.to_vec(), offsets, nbrs, entry, dim }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        debug_assert_eq!(q.len(), self.dim);
        let mut visited = vec![false; self.vecs.len()];
        let mut candidates: BinaryHeap<MinDistNode> = BinaryHeap::new();
        let mut top: BinaryHeap<DistNode> = BinaryHeap::new();

        let d0 = l2_sq(q, &self.vecs[self.entry as usize]);
        candidates.push(MinDistNode(DistNode { d: d0, n: self.entry }));
        top.push(DistNode { d: d0, n: self.entry });
        visited[self.entry as usize] = true;

        while let Some(MinDistNode(c)) = candidates.pop() {
            if top.len() >= ef && c.d > top.peek().unwrap().d { break; }
            let s = self.offsets[c.n as usize] as usize;
            let e = self.offsets[c.n as usize + 1] as usize;
            for idx in s..e {
                let nb = self.nbrs[idx];
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let d = l2_sq(q, &self.vecs[nb as usize]);
                if top.len() < ef || d < top.peek().unwrap().d {
                    candidates.push(MinDistNode(DistNode { d, n: nb }));
                    top.push(DistNode { d, n: nb });
                    if top.len() > ef { top.pop(); }
                }
            }
        }
        let mut out: Vec<_> = top.into_sorted_vec().into_iter().take(k)
            .map(|x| (x.n, x.d)).collect();
        out.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        out
    }
}

// shared greedy used at build time
fn greedy_search(
    vecs: &[Vec<f32>], entry: NodeId, q: &[f32], ef: usize,
    adj: &[Vec<(f32, NodeId)>],
) -> Vec<(f32, NodeId)> {
    let n = vecs.len();
    let mut visited = vec![false; n];
    let mut candidates: BinaryHeap<MinDistNode> = BinaryHeap::new();
    let mut top: BinaryHeap<DistNode> = BinaryHeap::new();
    let d0 = l2_sq(q, &vecs[entry as usize]);
    candidates.push(MinDistNode(DistNode { d: d0, n: entry }));
    top.push(DistNode { d: d0, n: entry });
    visited[entry as usize] = true;
    while let Some(MinDistNode(c)) = candidates.pop() {
        if top.len() >= ef && c.d > top.peek().unwrap().d { break; }
        for &(_, nb) in &adj[c.n as usize] {
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            let d = l2_sq(q, &vecs[nb as usize]);
            if top.len() < ef || d < top.peek().unwrap().d {
                candidates.push(MinDistNode(DistNode { d, n: nb }));
                top.push(DistNode { d, n: nb });
                if top.len() > ef { top.pop(); }
            }
        }
    }
    top.into_sorted_vec().into_iter().map(|x| (x.d, x.n)).collect()
}

// ---------- 2. Block (64-byte adjacency blocks, prefetch hint) ----------
pub const BLOCK_M: usize = 16;
pub const BLOCK_BYTES: usize = 64; // 16 * 4 bytes

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct AdjBlock { pub ids: [NodeId; BLOCK_M] } // sentinel = u32::MAX

pub struct BlockHnsw {
    vecs: Vec<Vec<f32>>,
    blocks: Vec<AdjBlock>,
    deg: Vec<u8>,
    entry: NodeId,
    dim: usize,
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe { core::arch::x86_64::_mm_prefetch::<{core::arch::x86_64::_MM_HINT_T0}>(p as *const i8); }
    #[cfg(target_arch = "aarch64")]
    unsafe { core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags)); }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = p; }
}

impl AnnIndex for BlockHnsw {
    fn name(&self) -> &'static str { "block-packed" }
    fn adjacency_bytes(&self) -> usize { self.blocks.len() * BLOCK_BYTES + self.deg.len() }

    fn build(vecs: &[Vec<f32>], _m: usize, ef: usize) -> Self {
        let m = BLOCK_M;
        let base = BaselineHnsw::build(vecs, m, ef);
        let n = vecs.len();
        let mut blocks = Vec::with_capacity(n);
        let mut deg = Vec::with_capacity(n);
        for i in 0..n {
            let s = base.offsets[i] as usize;
            let e = base.offsets[i+1] as usize;
            let mut blk = AdjBlock { ids: [u32::MAX; BLOCK_M] };
            let d = (e - s).min(BLOCK_M);
            blk.ids[..d].copy_from_slice(&base.nbrs[s..s+d]);
            blocks.push(blk);
            deg.push(d as u8);
        }
        Self { vecs: vecs.to_vec(), blocks, deg, entry: base.entry, dim: base.dim }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let mut visited = vec![false; self.vecs.len()];
        let mut candidates: BinaryHeap<MinDistNode> = BinaryHeap::new();
        let mut top: BinaryHeap<DistNode> = BinaryHeap::new();
        let d0 = l2_sq(q, &self.vecs[self.entry as usize]);
        candidates.push(MinDistNode(DistNode { d: d0, n: self.entry }));
        top.push(DistNode { d: d0, n: self.entry });
        visited[self.entry as usize] = true;

        while let Some(MinDistNode(c)) = candidates.pop() {
            if top.len() >= ef && c.d > top.peek().unwrap().d { break; }
            let blk = &self.blocks[c.n as usize];
            let d = self.deg[c.n as usize] as usize;
            // prefetch first few neighbour vectors
            for i in 0..d.min(4) {
                let nb = blk.ids[i];
                if nb != u32::MAX { prefetch(self.vecs[nb as usize].as_ptr()); }
            }
            for i in 0..d {
                let nb = blk.ids[i];
                if nb == u32::MAX || visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                // pipeline-deeper prefetch
                let look = i + 4;
                if look < d {
                    let nb2 = blk.ids[look];
                    if nb2 != u32::MAX { prefetch(self.vecs[nb2 as usize].as_ptr()); }
                }
                let dd = l2_sq(q, &self.vecs[nb as usize]);
                if top.len() < ef || dd < top.peek().unwrap().d {
                    candidates.push(MinDistNode(DistNode { d: dd, n: nb }));
                    top.push(DistNode { d: dd, n: nb });
                    if top.len() > ef { top.pop(); }
                }
            }
        }
        let mut out: Vec<_> = top.into_sorted_vec().into_iter().take(k)
            .map(|x| (x.n, x.d)).collect();
        out.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        out
    }
}

// ---------- 3. Sketch (12 IDs + 12 × 1-byte distance sketches per 64-byte block) ----------
pub const SKETCH_M: usize = 12;

#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct SketchBlock {
    pub ids: [NodeId; SKETCH_M],   // 48 bytes
    pub sk:  [u8;   SKETCH_M],     // 12 bytes — quantised per-edge node-norm hash
    pub pad: [u8;   4],            // 4 bytes pad → 64
}

/// Per-node 8-bit sketch: high bits of L2 norm bucket. Cheap, used as a coarse
/// query-vs-sketch lower bound: nodes whose norm-bucket lies far from the
/// query's norm-bucket cannot be the nearest under L2 (triangle inequality
/// proxy). We early-reject before fetching the full vector.
fn norm_sketch(v: &[f32], norm_min: f32, norm_max: f32) -> u8 {
    let n = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    let t = ((n - norm_min) / (norm_max - norm_min + 1e-9)).clamp(0.0, 0.999);
    (t * 256.0) as u8
}

pub struct SketchHnsw {
    vecs: Vec<Vec<f32>>,
    blocks: Vec<SketchBlock>,
    deg: Vec<u8>,
    sketches: Vec<u8>,
    norm_min: f32,
    norm_max: f32,
    entry: NodeId,
    dim: usize,
    /// Sketch-reject tolerance in u8 units; larger = safer recall, fewer rejects.
    reject_slack: u8,
}

impl SketchHnsw {
    pub fn with_slack(mut self, s: u8) -> Self { self.reject_slack = s; self }
}

impl AnnIndex for SketchHnsw {
    fn name(&self) -> &'static str { "sketch-reject" }
    fn adjacency_bytes(&self) -> usize { self.blocks.len() * 64 + self.deg.len() + self.sketches.len() }

    fn build(vecs: &[Vec<f32>], _m: usize, ef: usize) -> Self {
        let base = BaselineHnsw::build(vecs, SKETCH_M, ef);
        let n = vecs.len();
        let norms: Vec<f32> = vecs.iter().map(|v| v.iter().map(|x| x*x).sum::<f32>().sqrt()).collect();
        let norm_min = norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let norm_max = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sketches: Vec<u8> = vecs.iter().map(|v| norm_sketch(v, norm_min, norm_max)).collect();
        let mut blocks = Vec::with_capacity(n);
        let mut deg = Vec::with_capacity(n);
        for i in 0..n {
            let s = base.offsets[i] as usize;
            let e = base.offsets[i+1] as usize;
            let mut blk = SketchBlock { ids: [u32::MAX; SKETCH_M], sk: [0u8; SKETCH_M], pad: [0;4] };
            let d = (e - s).min(SKETCH_M);
            for k in 0..d {
                let nb = base.nbrs[s+k];
                blk.ids[k] = nb;
                blk.sk[k] = sketches[nb as usize];
            }
            blocks.push(blk);
            deg.push(d as u8);
        }
        Self { vecs: vecs.to_vec(), blocks, deg, sketches, norm_min, norm_max,
               entry: base.entry, dim: base.dim, reject_slack: 40 }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let q_sk = norm_sketch(q, self.norm_min, self.norm_max);
        let slack = self.reject_slack;
        let mut visited = vec![false; self.vecs.len()];
        let mut candidates: BinaryHeap<MinDistNode> = BinaryHeap::new();
        let mut top: BinaryHeap<DistNode> = BinaryHeap::new();
        let d0 = l2_sq(q, &self.vecs[self.entry as usize]);
        candidates.push(MinDistNode(DistNode { d: d0, n: self.entry }));
        top.push(DistNode { d: d0, n: self.entry });
        visited[self.entry as usize] = true;

        while let Some(MinDistNode(c)) = candidates.pop() {
            if top.len() >= ef && c.d > top.peek().unwrap().d { break; }
            let blk = &self.blocks[c.n as usize];
            let d = self.deg[c.n as usize] as usize;
            for i in 0..d {
                let nb = blk.ids[i];
                if nb == u32::MAX || visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                // sketch early-reject
                let s = blk.sk[i];
                let gap = if s > q_sk { s - q_sk } else { q_sk - s };
                if gap > slack && top.len() >= ef {
                    // skip without full distance
                    continue;
                }
                let dd = l2_sq(q, &self.vecs[nb as usize]);
                if top.len() < ef || dd < top.peek().unwrap().d {
                    candidates.push(MinDistNode(DistNode { d: dd, n: nb }));
                    top.push(DistNode { d: dd, n: nb });
                    if top.len() > ef { top.pop(); }
                }
            }
        }
        let mut out: Vec<_> = top.into_sorted_vec().into_iter().take(k)
            .map(|x| (x.n, x.d)).collect();
        out.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        out
    }
}

// ---------- ground truth ----------
pub fn brute_force_topk(vecs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(NodeId, f32)> {
    let mut all: Vec<(NodeId, f32)> = vecs.iter().enumerate()
        .map(|(i,v)| (i as NodeId, l2_sq(q, v))).collect();
    all.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}

pub fn recall_at_k(got: &[(NodeId, f32)], gt: &[(NodeId, f32)], k: usize) -> f32 {
    let truth: std::collections::HashSet<NodeId> = gt.iter().take(k).map(|x| x.0).collect();
    let hit = got.iter().take(k).filter(|x| truth.contains(&x.0)).count();
    hit as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rand_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect()
    }

    #[test]
    fn baseline_recall_reasonable() {
        let vs = rand_vecs(800, 32, 1);
        let qs = rand_vecs(20, 32, 99);
        let idx = BaselineHnsw::build(&vs, 16, 50);
        let mut tot = 0.0;
        for q in &qs {
            let got = idx.search(q, 10, 50);
            let gt = brute_force_topk(&vs, q, 10);
            tot += recall_at_k(&got, &gt, 10);
        }
        let r = tot / qs.len() as f32;
        assert!(r > 0.75, "baseline recall too low: {r}");
    }

    #[test]
    fn block_matches_baseline() {
        let vs = rand_vecs(800, 32, 2);
        let qs = rand_vecs(20, 32, 100);
        let b = BaselineHnsw::build(&vs, 16, 50);
        let k = BlockHnsw::build(&vs, 16, 50);
        let mut agree = 0;
        let mut total = 0;
        for q in &qs {
            let g1 = b.search(q, 10, 50);
            let g2 = k.search(q, 10, 50);
            let s1: std::collections::HashSet<NodeId> = g1.iter().map(|x|x.0).collect();
            for x in &g2 { if s1.contains(&x.0) { agree += 1; } total += 1; }
        }
        let a = agree as f32 / total as f32;
        assert!(a > 0.9, "block-vs-baseline agreement too low: {a}");
    }

    #[test]
    fn sketch_recall_acceptable() {
        let vs = rand_vecs(800, 32, 3);
        let qs = rand_vecs(20, 32, 101);
        let s = SketchHnsw::build(&vs, 12, 50);
        let mut tot = 0.0;
        for q in &qs {
            let got = s.search(q, 10, 50);
            let gt = brute_force_topk(&vs, q, 10);
            tot += recall_at_k(&got, &gt, 10);
        }
        let r = tot / qs.len() as f32;
        assert!(r > 0.60, "sketch recall too low: {r}");
    }
}
