//! Partial L2 distance with early abandonment.
//!
//! Given a running threshold `tau` (typically the k-th best squared L2
//! distance discovered so far), we can abort a per-vector distance
//! computation the moment the partial accumulator exceeds `tau`. This is
//! a classic AESA / branch-and-bound trick that stays underused in
//! Rust vector-search libraries because a naive `for` loop with a
//! per-lane branch kills auto-vectorization.
//!
//! This crate ships a small set of kernels that trade granularity of
//! the threshold check against SIMD throughput. All kernels operate on
//! `&[f32]` slices and are stable-Rust only — auto-vectorization is
//! driven by fixed-size chunk unrolling and `#[inline(always)]`.
//!
//! ## Kernels
//!
//! * [`l2_full`] — reference implementation. No branch inside the inner
//!   loop; the compiler emits packed SIMD on x86_64 (SSE/AVX) and
//!   aarch64 (NEON).
//! * [`l2_early_abandon`] — computes L2 in fixed-size blocks of `STRIDE`
//!   dimensions. After each block, compares the running sum against
//!   `tau_sq`. Returns [`AbandonResult::Abandoned`] as soon as the
//!   bound is exceeded, otherwise [`AbandonResult::Full`] with the
//!   final squared distance.
//! * [`l2_early_abandon_dyn`] — same as above but with a runtime stride.
//!
//! ## Threshold semantics
//!
//! All kernels operate on **squared** L2 distances. Callers that keep a
//! max-heap of the top-k must compare against the *squared* current
//! worst distance to avoid a `sqrt` in the inner loop.
//!
//! ## Ordering & drift
//!
//! Early abandon is deterministic given the input and threshold: two
//! points with the same coordinates will always return the same
//! `AbandonResult`. In particular, the recall of a rerank stage is
//! **not affected** — a vector is abandoned only when its true
//! distance is already known to exceed `tau_sq`, so it could never
//! have entered the top-k anyway.

#![deny(missing_docs)]

/// Result of a partial L2 computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbandonResult {
    /// The full distance was computed. Value is squared L2.
    Full(f32),
    /// Computation aborted early. Value is the partial squared
    /// accumulator at the point of the check that tripped `tau_sq`;
    /// callers should treat it as `>= tau_sq`.
    Abandoned(f32),
}

impl AbandonResult {
    /// Returns the value if `Full`, else `None`.
    #[inline]
    pub fn full(self) -> Option<f32> {
        match self {
            AbandonResult::Full(v) => Some(v),
            AbandonResult::Abandoned(_) => None,
        }
    }

    /// Returns true if the computation was abandoned early.
    #[inline]
    pub fn is_abandoned(self) -> bool {
        matches!(self, AbandonResult::Abandoned(_))
    }
}

/// Full squared L2 distance. Reference kernel, no early exit.
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn l2_full(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_full: length mismatch");
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }
    let mut acc = acc0 + acc1 + acc2 + acc3;
    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Partial squared L2 with early abandonment, checkpointing every
/// `STRIDE` dimensions. `STRIDE` must be a multiple of 4 for maximum
/// SIMD friendliness; the caller picks the trade-off.
///
/// Returns [`AbandonResult::Abandoned`] the first time the running
/// accumulator exceeds `tau_sq`, or [`AbandonResult::Full`] with the
/// final squared distance.
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn l2_early_abandon<const STRIDE: usize>(a: &[f32], b: &[f32], tau_sq: f32) -> AbandonResult {
    assert_eq!(a.len(), b.len(), "l2_early_abandon: length mismatch");
    debug_assert!(STRIDE >= 4 && STRIDE % 4 == 0, "STRIDE must be >=4 and multiple of 4");

    let dim = a.len();
    let mut acc = 0.0f32;
    let mut off = 0usize;

    while off + STRIDE <= dim {
        // Inner block: four accumulators to give the vectorizer a
        // dependency-free reduction tree.
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let sa = &a[off..off + STRIDE];
        let sb = &b[off..off + STRIDE];
        let inner = STRIDE / 4;
        for i in 0..inner {
            let j = i * 4;
            let d0 = sa[j] - sb[j];
            let d1 = sa[j + 1] - sb[j + 1];
            let d2 = sa[j + 2] - sb[j + 2];
            let d3 = sa[j + 3] - sb[j + 3];
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
        }
        acc += a0 + a1 + a2 + a3;
        if acc > tau_sq {
            return AbandonResult::Abandoned(acc);
        }
        off += STRIDE;
    }

    // Tail
    while off < dim {
        let d = a[off] - b[off];
        acc += d * d;
        off += 1;
    }

    AbandonResult::Full(acc)
}

/// Runtime-stride variant of [`l2_early_abandon`]. Useful when the
/// stride is chosen based on `dim` at runtime. Slightly slower than
/// the const-generic form because the inner block cannot be fully
/// unrolled by the optimizer.
#[inline]
pub fn l2_early_abandon_dyn(a: &[f32], b: &[f32], stride: usize, tau_sq: f32) -> AbandonResult {
    assert_eq!(a.len(), b.len(), "l2_early_abandon_dyn: length mismatch");
    assert!(stride >= 4 && stride % 4 == 0, "stride must be >=4 and multiple of 4");

    let dim = a.len();
    let mut acc = 0.0f32;
    let mut off = 0usize;

    while off + stride <= dim {
        let end = off + stride;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut i = off;
        while i + 4 <= end {
            let d0 = a[i] - b[i];
            let d1 = a[i + 1] - b[i + 1];
            let d2 = a[i + 2] - b[i + 2];
            let d3 = a[i + 3] - b[i + 3];
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
            i += 4;
        }
        acc += a0 + a1 + a2 + a3;
        if acc > tau_sq {
            return AbandonResult::Abandoned(acc);
        }
        off = end;
    }

    while off < dim {
        let d = a[off] - b[off];
        acc += d * d;
        off += 1;
    }

    AbandonResult::Full(acc)
}

/// A simple max-heap of size k used for top-k tracking during a rerank
/// scan. Kept here so the crate is self-contained and so benchmarks
/// can mirror how a real rerank stage would consume the kernels.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    /// (distance, id). Root is the current worst distance.
    heap: Vec<(f32, u32)>,
}

impl TopK {
    /// New empty heap with capacity `k`.
    pub fn new(k: usize) -> Self {
        Self { k, heap: Vec::with_capacity(k + 1) }
    }

    /// Current worst-in-top-k squared distance, or +inf if not yet full.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            // heap[0] is the max
            self.heap[0].0
        }
    }

    /// Try to insert `(dist_sq, id)`. Returns true if it entered the top-k.
    pub fn push(&mut self, dist_sq: f32, id: u32) -> bool {
        if self.heap.len() < self.k {
            self.heap.push((dist_sq, id));
            sift_up(&mut self.heap);
            true
        } else if dist_sq < self.heap[0].0 {
            self.heap[0] = (dist_sq, id);
            sift_down(&mut self.heap);
            true
        } else {
            false
        }
    }

    /// Consume and return the top-k sorted ascending by distance.
    pub fn into_sorted(mut self) -> Vec<(f32, u32)> {
        self.heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.heap
    }
}

fn sift_up(h: &mut [(f32, u32)]) {
    let mut i = h.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if h[i].0 > h[p].0 {
            h.swap(i, p);
            i = p;
        } else {
            break;
        }
    }
}

fn sift_down(h: &mut [(f32, u32)]) {
    let n = h.len();
    let mut i = 0usize;
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut m = i;
        if l < n && h[l].0 > h[m].0 {
            m = l;
        }
        if r < n && h[r].0 > h[m].0 {
            m = r;
        }
        if m == i {
            break;
        }
        h.swap(i, m);
        i = m;
    }
}

/// Rerank a candidate list against a query using full L2. Baseline.
pub fn rerank_full(query: &[f32], vectors: &[Vec<f32>], candidates: &[u32], k: usize) -> Vec<(f32, u32)> {
    let mut top = TopK::new(k);
    for &c in candidates {
        let d = l2_full(query, &vectors[c as usize]);
        top.push(d, c);
    }
    top.into_sorted()
}

/// Rerank a candidate list using const-stride early abandonment. Recall
/// is identical to [`rerank_full`] by construction: abandonment only
/// fires when `acc > threshold`, at which point the vector can never
/// enter the top-k.
pub fn rerank_early_abandon<const STRIDE: usize>(
    query: &[f32],
    vectors: &[Vec<f32>],
    candidates: &[u32],
    k: usize,
) -> Vec<(f32, u32)> {
    let mut top = TopK::new(k);
    for &c in candidates {
        let tau = top.threshold();
        match l2_early_abandon::<STRIDE>(query, &vectors[c as usize], tau) {
            AbandonResult::Full(d) => {
                top.push(d, c);
            }
            AbandonResult::Abandoned(_) => {}
        }
    }
    top.into_sorted()
}

/// Runtime-stride version of [`rerank_early_abandon`].
pub fn rerank_early_abandon_dyn(
    query: &[f32],
    vectors: &[Vec<f32>],
    candidates: &[u32],
    k: usize,
    stride: usize,
) -> Vec<(f32, u32)> {
    let mut top = TopK::new(k);
    for &c in candidates {
        let tau = top.threshold();
        match l2_early_abandon_dyn(query, &vectors[c as usize], stride, tau) {
            AbandonResult::Full(d) => {
                top.push(d, c);
            }
            AbandonResult::Abandoned(_) => {}
        }
    }
    top.into_sorted()
}

/// Estimated bytes of extra state per query for the early-abandon
/// rerank path. Currently just the `TopK` heap.
pub const fn per_query_overhead_bytes(k: usize) -> usize {
    // 12 bytes per (f32, u32) plus a Vec header (24 bytes on 64-bit).
    k * 12 + 24
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(seed: u64, dim: usize) -> Vec<f32> {
        // xorshift-style deterministic generator so tests don't drag rand.
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // map to [-1, 1]
                ((s as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn full_matches_reference() {
        let a = v(1, 128);
        let b = v(2, 128);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let got = l2_full(&a, &b);
        assert!((expected - got).abs() < 1e-3, "expected={expected} got={got}");
    }

    #[test]
    fn early_abandon_agrees_when_not_abandoned() {
        let a = v(3, 256);
        let b = v(4, 256);
        let full = l2_full(&a, &b);
        // Give a threshold well above the true distance.
        let r = l2_early_abandon::<32>(&a, &b, full * 2.0);
        match r {
            AbandonResult::Full(v) => {
                assert!((v - full).abs() < 1e-3, "expected {full} got {v}");
            }
            AbandonResult::Abandoned(_) => panic!("should not abandon"),
        }
    }

    #[test]
    fn early_abandon_abandons_when_threshold_low() {
        let a = v(5, 256);
        let b = v(6, 256);
        // Threshold = 0 forces immediate abandonment.
        let r = l2_early_abandon::<32>(&a, &b, 0.0);
        assert!(r.is_abandoned());
    }

    #[test]
    fn dyn_stride_matches_const_stride() {
        let a = v(7, 384);
        let b = v(8, 384);
        let full = l2_full(&a, &b);
        let c = l2_early_abandon::<64>(&a, &b, full * 2.0);
        let d = l2_early_abandon_dyn(&a, &b, 64, full * 2.0);
        assert_eq!(c, d);
    }

    #[test]
    fn topk_maintains_k_smallest() {
        let mut t = TopK::new(3);
        for (i, &d) in [10.0f32, 3.0, 7.0, 1.0, 8.0, 2.0].iter().enumerate() {
            t.push(d, i as u32);
        }
        let sorted = t.into_sorted();
        let dists: Vec<f32> = sorted.iter().map(|x| x.0).collect();
        assert_eq!(dists, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn rerank_recall_is_lossless() {
        // Recall parity: rerank_early_abandon must return the same set
        // of ids as rerank_full for any stride.
        let dim = 128;
        let n = 500;
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| v(1000 + i as u64, dim)).collect();
        let q = v(42, dim);
        let cands: Vec<u32> = (0..n as u32).collect();
        let k = 10;
        let full = rerank_full(&q, &vectors, &cands, k);
        for stride in [16usize, 32, 64, 128] {
            let ea = rerank_early_abandon_dyn(&q, &vectors, &cands, k, stride);
            let full_ids: Vec<u32> = full.iter().map(|x| x.1).collect();
            let ea_ids: Vec<u32> = ea.iter().map(|x| x.1).collect();
            assert_eq!(full_ids, ea_ids, "recall mismatch at stride={stride}");
            for (a, b) in full.iter().zip(&ea) {
                assert!((a.0 - b.0).abs() < 1e-3);
            }
        }
    }
}
