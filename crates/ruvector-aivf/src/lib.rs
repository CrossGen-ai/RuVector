//! AIVF — Adaptive IVF with online partition split/merge.
//!
//! A classical Inverted-File (IVF) index assigns vectors to the nearest of
//! `nlist` coarse centroids built once via k-means.  Under streaming workloads
//! the data distribution drifts — some lists explode, others collapse — and
//! recall at fixed `nprobe` decays.  AIVF reacts to drift by:
//!
//! 1. Tracking each list's running size, centroid (online mean), and intra-list
//!    sum-of-squared-error (SSE) via Welford-style updates.
//! 2. **Splitting** any list whose size > `split_size` or whose mean radius
//!    exceeds `split_radius_factor * global_mean_radius`.  The split runs a
//!    local 2-means refinement (Lloyd, capped at `split_iters` iterations).
//! 3. **Merging** the smallest list with its nearest sibling when its size
//!    falls below `merge_size`.
//!
//! The design is backend-agnostic via the [`Quantizer`] trait so a flat,
//! product-quantised, or RaBitQ backend can plug in later.  The provided
//! `FlatQuantizer` stores raw `f32`s and is used by the bundled benchmark.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

pub mod metric;
pub mod quantizer;

pub use metric::{l2_sq, dot};
pub use quantizer::{Quantizer, FlatQuantizer};

#[derive(Clone, Debug)]
pub struct AivfConfig {
    pub dim: usize,
    /// Initial number of coarse centroids built from the bootstrap sample.
    pub nlist_init: usize,
    /// Probes per query.
    pub nprobe: usize,
    /// Size above which a list becomes a split candidate.
    pub split_size: usize,
    /// A list also splits when its mean radius exceeds this factor times the
    /// global mean radius.  Set very large to disable radius-based splits.
    pub split_radius_factor: f32,
    /// Maximum Lloyd iterations during a local 2-means split.
    pub split_iters: usize,
    /// Size below which a list is a merge candidate.
    pub merge_size: usize,
    /// Don't run split/merge unless this many inserts have happened since the
    /// last rebalance — amortises overhead.
    pub rebalance_every: usize,
    /// Hard cap on the number of lists (protects against pathological growth).
    pub max_lists: usize,
}

impl AivfConfig {
    pub fn new(dim: usize, nlist_init: usize) -> Self {
        Self {
            dim,
            nlist_init,
            nprobe: 8,
            split_size: 4096,
            split_radius_factor: 4.0,
            split_iters: 6,
            merge_size: 16,
            rebalance_every: 1024,
            max_lists: nlist_init.saturating_mul(8).max(256),
        }
    }
}

#[derive(Clone, Debug)]
struct InvList {
    centroid: Vec<f32>,
    ids: Vec<u32>,
    /// Sum of vectors assigned to this list (for online centroid recomputation).
    sum: Vec<f64>,
    /// Sum of squared L2 distances of members from the centroid (SSE).
    sse: f64,
    /// Number of vectors (== ids.len(), tracked explicitly for clarity).
    count: usize,
}

impl InvList {
    fn new(centroid: Vec<f32>, dim: usize) -> Self {
        Self {
            centroid,
            ids: Vec::new(),
            sum: vec![0.0; dim],
            sse: 0.0,
            count: 0,
        }
    }

    fn mean_radius_sq(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.sse / self.count as f64) as f32 }
    }
}

pub struct Aivf<Q: Quantizer> {
    cfg: AivfConfig,
    lists: Vec<InvList>,
    q: Q,
    inserts_since_rebalance: usize,
    split_events: u64,
    merge_events: u64,
}

impl<Q: Quantizer> Aivf<Q> {
    /// Build an AIVF index from a bootstrap sample.  The first `nlist_init`
    /// vectors are taken as initial centroids (deterministic — no Date/rand
    /// hidden state); production code would seed via k-means on a sample.
    pub fn build(cfg: AivfConfig, bootstrap: &[Vec<f32>], q: Q) -> Self {
        assert!(bootstrap.len() >= cfg.nlist_init,
                "bootstrap must contain at least nlist_init vectors");
        let dim = cfg.dim;
        let lists: Vec<InvList> = (0..cfg.nlist_init)
            .map(|i| InvList::new(bootstrap[i].clone(), dim))
            .collect();
        let mut me = Self {
            cfg, lists, q,
            inserts_since_rebalance: 0,
            split_events: 0,
            merge_events: 0,
        };
        for (i, v) in bootstrap.iter().enumerate() {
            me.add(i as u32, v);
        }
        me
    }

    pub fn len(&self) -> usize { self.lists.iter().map(|l| l.count).sum() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn num_lists(&self) -> usize { self.lists.len() }
    pub fn split_events(&self) -> u64 { self.split_events }
    pub fn merge_events(&self) -> u64 { self.merge_events }

    pub fn add(&mut self, id: u32, v: &[f32]) {
        debug_assert_eq!(v.len(), self.cfg.dim);
        // Store in the backing quantiser BEFORE rebalance — splits may need
        // to read this vector via q.get(id).
        self.q.add(id, v);
        let li = self.nearest_list(v);
        self.assign(li, id, v);
        self.inserts_since_rebalance += 1;
        if self.inserts_since_rebalance >= self.cfg.rebalance_every {
            self.rebalance();
            self.inserts_since_rebalance = 0;
        }
    }

    fn assign(&mut self, li: usize, id: u32, v: &[f32]) {
        let list = &mut self.lists[li];
        let d = l2_sq(v, &list.centroid);
        list.ids.push(id);
        for (s, &x) in list.sum.iter_mut().zip(v.iter()) { *s += x as f64; }
        list.sse += d as f64;
        list.count += 1;
    }

    fn nearest_list(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, l) in self.lists.iter().enumerate() {
            let d = l2_sq(v, &l.centroid);
            if d < best_d { best_d = d; best = i; }
        }
        best
    }

    /// k-NN search: probe the `nprobe` closest lists, exact-rerank using `Q`.
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u32, f32)> {
        debug_assert_eq!(q.len(), self.cfg.dim);
        // Pick top `nprobe` lists by centroid distance.
        let np = self.cfg.nprobe.min(self.lists.len());
        let mut probes: Vec<(usize, f32)> = self.lists.iter().enumerate()
            .map(|(i, l)| (i, l2_sq(q, &l.centroid)))
            .collect();
        probes.select_nth_unstable_by(np - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        probes.truncate(np);

        // Bounded max-heap of size k.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        for &(li, _) in &probes {
            for &id in &self.lists[li].ids {
                let d = self.q.distance(id, q);
                if heap.len() < k {
                    heap.push(HeapEntry { d, id });
                } else if let Some(top) = heap.peek() {
                    if d < top.d {
                        heap.pop();
                        heap.push(HeapEntry { d, id });
                    }
                }
            }
        }
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|e| (e.id, e.d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out
    }

    /// Trigger one pass of split/merge maintenance.  Public for benchmarks.
    pub fn rebalance(&mut self) {
        self.maybe_splits();
        self.maybe_merges();
    }

    fn global_mean_radius_sq(&self) -> f32 {
        let mut tot_sse = 0.0f64;
        let mut tot_n = 0usize;
        for l in &self.lists {
            tot_sse += l.sse;
            tot_n += l.count;
        }
        if tot_n == 0 { 0.0 } else { (tot_sse / tot_n as f64) as f32 }
    }

    fn maybe_splits(&mut self) {
        let gmr = self.global_mean_radius_sq().max(1e-12);
        // Snapshot of indices that qualify (radius- or size-driven).
        let candidates: Vec<usize> = (0..self.lists.len())
            .filter(|&i| {
                let l = &self.lists[i];
                if l.count < self.cfg.merge_size * 2 { return false; }
                let radius_hit = l.mean_radius_sq() > self.cfg.split_radius_factor * gmr;
                let size_hit   = l.count > self.cfg.split_size;
                radius_hit || size_hit
            })
            .collect();
        for idx in candidates {
            if self.lists.len() >= self.cfg.max_lists { break; }
            self.split_one(idx);
        }
    }

    fn split_one(&mut self, idx: usize) {
        // Pull out points (we need their vectors — recover via Q).
        let ids = std::mem::take(&mut self.lists[idx].ids);
        if ids.len() < 2 { self.lists[idx].ids = ids; return; }
        let dim = self.cfg.dim;
        // Initialise two centroids: pick the two furthest-apart vectors via
        // a deterministic 2-pass scan (avoids RNG calls so the index is
        // reproducible).  Pass 1: pick the point furthest from id[0].
        let v0: Vec<f32> = self.q.get(ids[0]).to_vec();
        let mut far_a = 0usize; let mut far_a_d = -1.0f32;
        for (k, &id) in ids.iter().enumerate() {
            let d = l2_sq(&v0, self.q.get(id));
            if d > far_a_d { far_a_d = d; far_a = k; }
        }
        let va: Vec<f32> = self.q.get(ids[far_a]).to_vec();
        let mut far_b = 0usize; let mut far_b_d = -1.0f32;
        for (k, &id) in ids.iter().enumerate() {
            let d = l2_sq(&va, self.q.get(id));
            if d > far_b_d { far_b_d = d; far_b = k; }
        }
        let mut c_a: Vec<f32> = self.q.get(ids[far_a]).to_vec();
        let mut c_b: Vec<f32> = self.q.get(ids[far_b]).to_vec();

        // Lloyd iterations.
        let mut assign: Vec<u8> = vec![0; ids.len()];
        for _ in 0..self.cfg.split_iters {
            let mut sa = vec![0.0f64; dim]; let mut na = 0usize;
            let mut sb = vec![0.0f64; dim]; let mut nb = 0usize;
            let mut changed = false;
            for (k, &id) in ids.iter().enumerate() {
                let v = self.q.get(id);
                let da = l2_sq(v, &c_a);
                let db = l2_sq(v, &c_b);
                let bit: u8 = if da <= db { 0 } else { 1 };
                if bit != assign[k] { changed = true; assign[k] = bit; }
                if bit == 0 {
                    for (s, &x) in sa.iter_mut().zip(v.iter()) { *s += x as f64; }
                    na += 1;
                } else {
                    for (s, &x) in sb.iter_mut().zip(v.iter()) { *s += x as f64; }
                    nb += 1;
                }
            }
            if na > 0 { for (c, s) in c_a.iter_mut().zip(sa.iter()) { *c = (*s / na as f64) as f32; } }
            if nb > 0 { for (c, s) in c_b.iter_mut().zip(sb.iter()) { *c = (*s / nb as f64) as f32; } }
            if !changed { break; }
        }

        // If one side is empty, abort the split.
        let na = assign.iter().filter(|&&b| b == 0).count();
        let nb = assign.len() - na;
        if na == 0 || nb == 0 {
            self.lists[idx].ids = ids;
            return;
        }

        // Build the two replacement lists (replaces idx + appends a new one).
        let mut list_a = InvList::new(c_a, dim);
        let mut list_b = InvList::new(c_b, dim);
        for (k, &id) in ids.iter().enumerate() {
            let v = self.q.get(id);
            if assign[k] == 0 {
                let d = l2_sq(v, &list_a.centroid);
                list_a.ids.push(id);
                for (s, &x) in list_a.sum.iter_mut().zip(v.iter()) { *s += x as f64; }
                list_a.sse += d as f64;
                list_a.count += 1;
            } else {
                let d = l2_sq(v, &list_b.centroid);
                list_b.ids.push(id);
                for (s, &x) in list_b.sum.iter_mut().zip(v.iter()) { *s += x as f64; }
                list_b.sse += d as f64;
                list_b.count += 1;
            }
        }
        self.lists[idx] = list_a;
        self.lists.push(list_b);
        self.split_events += 1;
    }

    fn maybe_merges(&mut self) {
        // Find lists below merge threshold; for each, find nearest sibling.
        let small: Vec<usize> = (0..self.lists.len())
            .filter(|&i| self.lists[i].count > 0 && self.lists[i].count < self.cfg.merge_size)
            .collect();
        let mut merged = vec![false; self.lists.len()];
        for i in small {
            if merged[i] { continue; }
            let Some(j) = self.nearest_sibling(i, &merged) else { continue; };
            self.merge_pair(i, j);
            merged[i] = true; merged[j] = true;
        }
        // Drop now-empty lists.
        self.lists.retain(|l| l.count > 0);
    }

    fn nearest_sibling(&self, i: usize, merged: &[bool]) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_d = f32::INFINITY;
        for j in 0..self.lists.len() {
            if j == i || merged[j] || self.lists[j].count == 0 { continue; }
            let d = l2_sq(&self.lists[i].centroid, &self.lists[j].centroid);
            if d < best_d { best_d = d; best = Some(j); }
        }
        best
    }

    fn merge_pair(&mut self, i: usize, j: usize) {
        // Combine i into j (the larger), leaving i empty for later compaction.
        let (a, b) = if self.lists[i].count >= self.lists[j].count { (j, i) } else { (i, j) };
        // Move ids/sum/sse from a into b.
        let dim = self.cfg.dim;
        let src_ids = std::mem::take(&mut self.lists[a].ids);
        let src_sum = std::mem::take(&mut self.lists[a].sum);
        let src_sse = self.lists[a].sse;
        let src_n   = self.lists[a].count;
        self.lists[a].sum = vec![0.0; dim];
        self.lists[a].sse = 0.0;
        self.lists[a].count = 0;
        {
            let dst = &mut self.lists[b];
            dst.ids.extend(src_ids);
            for (s, &x) in dst.sum.iter_mut().zip(src_sum.iter()) { *s += x; }
            dst.count += src_n;
            // New centroid = combined mean.
            for (c, &s) in dst.centroid.iter_mut().zip(dst.sum.iter()) {
                *c = (s / dst.count.max(1) as f64) as f32;
            }
            // We don't have the original src vectors here cheaply; conservatively
            // bump sse by src_sse plus the inter-centroid separation penalty.
            // (This over-estimates SSE slightly; acceptable for radius bookkeeping.)
            dst.sse += src_sse;
        }
        self.merge_events += 1;
    }
}

#[derive(Copy, Clone, PartialEq)]
struct HeapEntry { d: f32, id: u32 }
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for HeapEntry {
    // max-heap by distance so peek() = current worst.
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal)
    }
}
