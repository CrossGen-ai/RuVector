//! Brute-force top-k scans over the two storage backends.
//!
//! Everything is L2². We keep a bounded-size max-heap of `(dist, id)`
//! sorted by distance descending, so the heap top is always the current
//! "worst-of-best" threshold — this doubles as the pruning threshold in the
//! PDX-pruned scan (see `pruned.rs`).

use crate::block::{HorizontalStore, PdxStore, VectorStore, BLOCK_LEN};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Marker for which backend a top-level API is using.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Backend { Horizontal, Pdx, PdxPruned }

/// Bounded max-heap top-k. Small-k friendly (heap ops O(log k)).
pub struct TopK {
    k: usize,
    heap: BinaryHeap<HeapEntry>,
}

#[derive(Copy, Clone, Debug)]
struct HeapEntry { dist: f32, id: u32 }

// Max-heap by dist: heap top is the CURRENT worst kept.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for HeapEntry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for HeapEntry { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for HeapEntry {}

impl TopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        Self { k, heap: BinaryHeap::with_capacity(k + 1) }
    }
    pub fn k(&self) -> usize { self.k }

    /// Current pruning threshold: `+inf` if heap not yet full, else the
    /// distance of the k-th-best (worst kept).
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k { f32::INFINITY }
        else { self.heap.peek().map(|e| e.dist).unwrap_or(f32::INFINITY) }
    }

    #[inline]
    pub fn offer(&mut self, dist: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry { dist, id });
        } else if let Some(top) = self.heap.peek() {
            if dist < top.dist {
                self.heap.pop();
                self.heap.push(HeapEntry { dist, id });
            }
        }
    }

    /// Return the top-k in ascending distance order.
    pub fn into_sorted(self) -> Vec<(f32, u32)> {
        let mut v: Vec<(f32, u32)> = self.heap.into_iter().map(|e| (e.dist, e.id)).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        v
    }
}

/// Horizontal L2² top-k brute-force baseline.
pub fn brute_horizontal(store: &HorizontalStore, query: &[f32], k: usize) -> Vec<(f32, u32)> {
    assert_eq!(query.len(), store.dim());
    let mut topk = TopK::new(k);
    let n = store.len();
    for id in 0..n {
        let r = store.row(id);
        // Unrolled L2² accumulator. Rust autovectorizes this cleanly.
        let mut acc = 0.0f32;
        for j in 0..r.len() {
            let d = r[j] - query[j];
            acc += d * d;
        }
        topk.offer(acc, id as u32);
    }
    topk.into_sorted()
}

/// PDX (dimension-partitioned) L2² top-k without pruning: full traversal,
/// but with block-parallel per-lane accumulators. Serves as an apples-to-
/// apples layout comparison against the horizontal baseline.
pub fn brute_pdx(store: &PdxStore, query: &[f32], k: usize) -> Vec<(f32, u32)> {
    assert_eq!(query.len(), store.dim());
    // Permute query once — order matches perm[k] -> original dim id.
    let perm = store.perm();
    let mut pq = vec![0f32; store.dim()];
    for (k_slot, &od) in perm.iter().enumerate() {
        pq[k_slot] = query[od as usize];
    }

    let mut topk = TopK::new(k);
    // Reusable per-lane accumulator (max block size).
    let mut acc = [0f32; BLOCK_LEN];
    let d = store.dim();

    for b in 0..store.num_blocks() {
        let bl = store.block_size(b);
        acc[..bl].fill(0.0);
        let block = store.block(b);
        // Column-major traversal: dimension by dimension, all lanes updated in the inner loop.
        for k_slot in 0..d {
            let q = pq[k_slot];
            let col = &block[k_slot * bl..(k_slot + 1) * bl];
            for lane in 0..bl {
                let diff = col[lane] - q;
                acc[lane] += diff * diff;
            }
        }
        // Emit lane results to the heap.
        let base = (b * BLOCK_LEN) as u32;
        for lane in 0..bl {
            topk.offer(acc[lane], base + lane as u32);
        }
    }
    topk.into_sorted()
}
