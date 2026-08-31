//! IVF posting-list scan variants.
//!
//! A "posting list" here is a contiguous `Vec<f32>` of length `n_vectors * dim`
//! (row-major). This is the layout FAISS's `IndexIVFFlat` uses inside a single
//! inverted list, and the layout HNSW-flat and DiskANN's SSD tier use inside a
//! sector. Contiguous row-major is what makes software prefetch pay off — the
//! next vector is exactly `dim * 4` bytes ahead in memory, so a single
//! `prefetch(ptr + dim * 4 * K)` covers `ceil(dim*4/64)` cache lines with one
//! instruction on Apple Silicon (M-series prefetches a full 128-byte line on
//! the load side).
//!
//! Three scan strategies expose the memory-vs-compute trade-off:
//!  1. `scan_no_prefetch`      — baseline
//!  2. `scan_fixed_prefetch`   — always prefetch `K` vectors ahead
//!  3. `scan_adaptive_prefetch` — pick `K` from vector byte size

use crate::distance::l2_sq;
use crate::prefetch::{adaptive_lookahead, prefetch_read, Locality};

/// A single result — (posting-list-local index, squared distance).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub idx: u32,
    pub dist_sq: f32,
}

impl Eq for Hit {}

/// Fixed-capacity top-k heap. Small-k is dominant for ANN so we keep it as a
/// linear-scan max-list rather than a binary heap — the branch predictor eats
/// this for lunch when k <= 32.
pub struct TopK {
    cap: usize,
    hits: Vec<Hit>,
    // Cached worst dist in `hits`. `f32::INFINITY` while under capacity.
    worst: f32,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            cap: k,
            hits: Vec::with_capacity(k),
            worst: f32::INFINITY,
        }
    }

    #[inline]
    pub fn worst_dist(&self) -> f32 {
        self.worst
    }

    #[inline]
    pub fn push(&mut self, hit: Hit) {
        if self.hits.len() < self.cap {
            self.hits.push(hit);
            if self.hits.len() == self.cap {
                // First time we filled — set `worst` to the largest.
                self.worst = self
                    .hits
                    .iter()
                    .map(|h| h.dist_sq)
                    .fold(f32::NEG_INFINITY, f32::max);
            }
            return;
        }
        if hit.dist_sq >= self.worst {
            return;
        }
        // Replace the current worst.
        let (worst_pos, _) = self
            .hits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.dist_sq.partial_cmp(&b.dist_sq).unwrap())
            .unwrap();
        self.hits[worst_pos] = hit;
        // Recompute worst.
        self.worst = self
            .hits
            .iter()
            .map(|h| h.dist_sq)
            .fold(f32::NEG_INFINITY, f32::max);
    }

    pub fn into_sorted(mut self) -> Vec<Hit> {
        self.hits
            .sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        self.hits
    }
}

/// A contiguous posting list.
pub struct PostingList {
    pub dim: usize,
    pub vectors: Vec<f32>, // len = n * dim
}

impl PostingList {
    pub fn new(dim: usize, vectors: Vec<f32>) -> Self {
        assert!(dim > 0);
        assert_eq!(vectors.len() % dim, 0, "vectors not a multiple of dim");
        Self { dim, vectors }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dim
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline]
    pub fn row_ptr(&self, i: usize) -> *const f32 {
        // SAFETY: caller passes `i < self.len()`. Callers are internal only.
        unsafe { self.vectors.as_ptr().add(i * self.dim) }
    }
}

/// Variant 1: baseline scan, no software prefetch. Relies purely on the
/// hardware stream prefetcher. On M4 Max that's competitive for dim=128
/// contiguous scans; the gap widens for larger dims where the reorder buffer
/// starts to stall.
pub fn scan_no_prefetch(list: &PostingList, query: &[f32], k: usize) -> Vec<Hit> {
    let mut top = TopK::new(k);
    let n = list.len();
    for i in 0..n {
        let d = l2_sq(list.row(i), query);
        top.push(Hit {
            idx: i as u32,
            dist_sq: d,
        });
    }
    top.into_sorted()
}

/// Variant 2: fixed lookahead prefetch. `K` vectors ahead of the current
/// distance computation, we prefetch the first cache line of the vector we
/// will read after that. Extra rows in the same vector get picked up by the
/// hardware prefetcher once we begin reading the first.
pub fn scan_fixed_prefetch(
    list: &PostingList,
    query: &[f32],
    k: usize,
    lookahead: usize,
) -> Vec<Hit> {
    let mut top = TopK::new(k);
    let n = list.len();
    let dim = list.dim;
    let bytes_per_vec = dim * core::mem::size_of::<f32>();
    let lines_per_vec = (bytes_per_vec + 63) / 64;

    for i in 0..n {
        // Prefetch vector `i + lookahead` before we compute for vector `i`.
        let target = i + lookahead;
        if target < n {
            let base = list.row_ptr(target);
            // Prefetch every cache line of the target vector. For dim=128
            // (512 bytes) that's 8 prefetches — cheap compared to the L2
            // miss they hide.
            for line in 0..lines_per_vec {
                // 16 f32s per 64-byte line.
                let offset = line * 16;
                // SAFETY: we're only asking the prefetcher to warm the line;
                // even if `offset` overshoots the vector by a few bytes into
                // the next vector, that next vector is either still valid
                // memory (same allocation) or, in the very last iteration,
                // clamped by `target < n` — but the tail spill within the
                // last vector is guaranteed valid since Vec<f32> allocates
                // in bulk.
                unsafe {
                    prefetch_read(base.add(offset), Locality::L1);
                }
            }
        }

        let d = l2_sq(list.row(i), query);
        top.push(Hit {
            idx: i as u32,
            dist_sq: d,
        });
    }
    top.into_sorted()
}

/// Variant 3: adaptive-lookahead prefetch. Same mechanism as Variant 2, but
/// the lookahead is picked from vector byte size so the same code auto-tunes
/// as `dim` changes.
pub fn scan_adaptive_prefetch(list: &PostingList, query: &[f32], k: usize) -> Vec<Hit> {
    let bytes_per_vec = list.dim * core::mem::size_of::<f32>();
    // 4 cache lines is the sweet spot on Apple Silicon in our measurements —
    // enough to cover memory latency for dim<=256, not so much that we
    // pollute L2.
    let lookahead = adaptive_lookahead(bytes_per_vec, 4);
    scan_fixed_prefetch(list, query, k, lookahead)
}

/// Variant 4: strided-order scan of a contiguous buffer, mimicking the
/// access pattern of a "gather" over an IVF-PQ index where the query touches
/// several posting lists whose vectors are interleaved by row-id rather than
/// contiguous. The hardware stream prefetcher cannot help here — every read
/// is effectively a cold cache line. This is the regime where software
/// prefetch is expected to pay off.
pub fn scan_strided_no_prefetch(
    list: &PostingList,
    query: &[f32],
    k: usize,
    stride: usize,
) -> Vec<Hit> {
    let mut top = TopK::new(k);
    let order = strided_order(list.len(), stride);
    for i in order {
        let d = l2_sq(list.row(i), query);
        top.push(Hit {
            idx: i as u32,
            dist_sq: d,
        });
    }
    top.into_sorted()
}

/// Variant 5: strided scan with software prefetch. `lookahead` positions
/// ahead in the strided order, we prefetch the entire target vector.
pub fn scan_strided_prefetch(
    list: &PostingList,
    query: &[f32],
    k: usize,
    stride: usize,
    lookahead: usize,
) -> Vec<Hit> {
    let mut top = TopK::new(k);
    let dim = list.dim;
    let bytes_per_vec = dim * core::mem::size_of::<f32>();
    let lines_per_vec = (bytes_per_vec + 63) / 64;
    let order: Vec<usize> = strided_order(list.len(), stride).collect();
    let n = order.len();

    for pos in 0..n {
        let target_pos = pos + lookahead;
        if target_pos < n {
            let target_idx = order[target_pos];
            let base = list.row_ptr(target_idx);
            for line in 0..lines_per_vec {
                let offset = line * 16;
                unsafe {
                    prefetch_read(base.add(offset), Locality::L1);
                }
            }
        }
        let i = order[pos];
        let d = l2_sq(list.row(i), query);
        top.push(Hit {
            idx: i as u32,
            dist_sq: d,
        });
    }
    top.into_sorted()
}

/// Emit indices in a strided order that visits every element exactly once.
/// If `gcd(stride, n) != 1` the cycle would short-circuit, so we fall back
/// to linear order for those cases — the caller can pick a coprime stride
/// (e.g. any prime relative to `n`).
fn strided_order(n: usize, stride: usize) -> impl Iterator<Item = usize> {
    let stride = if n == 0 || gcd(stride, n) != 1 {
        1
    } else {
        stride
    };
    (0..n).map(move |i| (i * stride) % n.max(1))
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_list() -> PostingList {
        // 6 vectors of dim 4.
        let v: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, 0.0, // 1
            0.0, 1.0, 0.0, 0.0, // 2
            0.0, 0.0, 1.0, 0.0, // 3
            0.0, 0.0, 0.0, 1.0, // 4
            2.0, 2.0, 2.0, 2.0, // 5
        ];
        PostingList::new(4, v)
    }

    #[test]
    fn topk_basic() {
        let mut t = TopK::new(3);
        for (i, d) in [5.0, 1.0, 9.0, 2.0, 4.0].iter().enumerate() {
            t.push(Hit {
                idx: i as u32,
                dist_sq: *d,
            });
        }
        let sorted = t.into_sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].dist_sq, 1.0);
        assert_eq!(sorted[1].dist_sq, 2.0);
        assert_eq!(sorted[2].dist_sq, 4.0);
    }

    #[test]
    fn all_variants_agree() {
        let list = tiny_list();
        let q = [0.1f32, 0.0, 0.0, 0.0];
        let k = 3;
        let a = scan_no_prefetch(&list, &q, k);
        let b = scan_fixed_prefetch(&list, &q, k, 2);
        let c = scan_adaptive_prefetch(&list, &q, k);
        assert_eq!(a, b, "fixed prefetch disagrees with baseline");
        assert_eq!(a, c, "adaptive prefetch disagrees with baseline");
        // Nearest to (0.1,0,0,0): row 1 (dist_sq=0.81), row 0 (0.01), row 2 (1.01)…
        // Actually nearest is row 0 at 0.01, row 1 at 0.81, row 2 at 1.01.
        assert_eq!(a[0].idx, 0);
        assert_eq!(a[1].idx, 1);
        assert_eq!(a[2].idx, 2);
    }

    #[test]
    fn strided_variants_agree() {
        // Distinct-distance list so tie-breaks don't turn into a spurious
        // failure between visit orders.
        let v: Vec<f32> = vec![
            0.0, 0.0, 0.0, // 0 -> dist^2 = 0.01
            1.0, 0.0, 0.0, // 1 -> 0.81
            0.0, 1.0, 0.0, // 2 -> 1.01
            0.0, 0.0, 2.0, // 3 -> 4.01
            0.0, 0.0, 3.0, // 4 -> 9.01
            5.0, 5.0, 5.0, // 5 -> ~99
            0.5, 0.0, 0.0, // 6 -> 0.16
        ];
        let list = PostingList::new(3, v);
        let q = [0.1f32, 0.0, 0.0];
        let base = scan_no_prefetch(&list, &q, 4);
        // stride=3 is coprime with n=7.
        let s = scan_strided_no_prefetch(&list, &q, 4, 3);
        let sp = scan_strided_prefetch(&list, &q, 4, 3, 2);
        // Same set of ids, same distances, sorted by dist so order matches.
        assert_eq!(base, s);
        assert_eq!(base, sp);
    }

    #[test]
    fn empty_list_returns_empty() {
        let list = PostingList::new(4, vec![]);
        let q = vec![0.0f32; 4];
        assert!(scan_no_prefetch(&list, &q, 5).is_empty());
        assert!(scan_fixed_prefetch(&list, &q, 5, 4).is_empty());
        assert!(scan_adaptive_prefetch(&list, &q, 5).is_empty());
    }
}
