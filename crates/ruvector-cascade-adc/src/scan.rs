//! Scanner trait + three concrete scanners.

use crate::codebook::{K4, K8};
use crate::pq::PqIndex;

/// A single result: id + estimated squared L2 distance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScanResult {
    pub id: u32,
    pub dist: f32,
}

/// Common interface for the three scanners.
pub trait Scanner {
    /// Query the index and return the top-`k` candidates.
    /// `out` is expected to have capacity `k`; it will be cleared and refilled.
    fn search(&self, index: &PqIndex, query: &[f32], k: usize, out: &mut Vec<ScanResult>);

    /// Human-readable name for benchmark reporting.
    fn name(&self) -> &'static str;
}

/// Baseline: scan every vector at 8-bit precision.
pub struct FullEightBitScanner;

impl Scanner for FullEightBitScanner {
    fn name(&self) -> &'static str { "full-8bit" }

    fn search(&self, index: &PqIndex, query: &[f32], k: usize, out: &mut Vec<ScanResult>) {
        let m = index.params.m;
        let lut = index.lut8(query);
        let mut heap: Vec<ScanResult> = Vec::with_capacity(k + 1);
        for id in 0..index.n {
            let code_off = id * m;
            let mut s = 0.0f32;
            for j in 0..m {
                let c = index.codes8[code_off + j] as usize;
                s += lut[j * K8 + c];
            }
            push_topk(&mut heap, k, ScanResult { id: id as u32, dist: s });
        }
        finalise(heap, out);
    }
}

/// Aggressive: scan every vector at 4-bit precision only.
pub struct FullFourBitScanner;

impl Scanner for FullFourBitScanner {
    fn name(&self) -> &'static str { "full-4bit" }

    fn search(&self, index: &PqIndex, query: &[f32], k: usize, out: &mut Vec<ScanResult>) {
        let m = index.params.m;
        let stride = m.div_ceil(2);
        let lut = index.lut4(query);
        let mut heap: Vec<ScanResult> = Vec::with_capacity(k + 1);
        // Unroll two codes/byte: avoids conditional branch on parity.
        let full_pairs = m / 2;
        let has_odd = m % 2 == 1;
        for id in 0..index.n {
            let code_off = id * stride;
            let mut s = 0.0f32;
            for p in 0..full_pairs {
                let byte = index.codes4[code_off + p];
                let lo = (byte & 0x0F) as usize;
                let hi = (byte >> 4) as usize;
                s += lut[(2*p) * K4 + lo] + lut[(2*p + 1) * K4 + hi];
            }
            if has_odd {
                let byte = index.codes4[code_off + full_pairs];
                let lo = (byte & 0x0F) as usize;
                s += lut[(2*full_pairs) * K4 + lo];
            }
            push_topk(&mut heap, k, ScanResult { id: id as u32, dist: s });
        }
        finalise(heap, out);
    }
}

/// Cascade: coarse 4-bit pass keeps top `rho * n` (or `top_t`), then 8-bit
/// refine on survivors.
pub struct CascadeScanner {
    /// Fraction of the corpus that survives to Stage 2 (0.0 < rho ≤ 1.0).
    pub rho: f32,
    /// Optional absolute floor on the survivor set size (in case `rho * n` is
    /// smaller than useful for a given `k`). If `Some`, the survivor count is
    /// `max(top_t, ceil(rho * n))`.
    pub top_t: Option<usize>,
}

impl CascadeScanner {
    pub fn new(rho: f32) -> Self {
        Self { rho, top_t: None }
    }
    pub fn with_floor(rho: f32, top_t: usize) -> Self {
        Self { rho, top_t: Some(top_t) }
    }
}

impl Scanner for CascadeScanner {
    fn name(&self) -> &'static str { "cascade" }

    fn search(&self, index: &PqIndex, query: &[f32], k: usize, out: &mut Vec<ScanResult>) {
        let m = index.params.m;
        let stride4 = m.div_ceil(2);
        let lut4 = index.lut4(query);

        // Stage 1: coarse scan; keep top T survivors.
        let mut t = ((self.rho * index.n as f32).ceil() as usize).max(k);
        if let Some(floor) = self.top_t {
            t = t.max(floor);
        }
        t = t.min(index.n);

        let full_pairs = m / 2;
        let has_odd = m % 2 == 1;
        // Flat-array pass: compute all N coarse distances, then use
        // select_nth_unstable to partition the top T without full sort.
        // This is O(N) vs O(N log T) for a size-T heap, and — critically —
        // avoids any branchy heap operation in the tight scan loop.
        let mut dists: Vec<ScanResult> = Vec::with_capacity(index.n);
        for id in 0..index.n {
            let code_off = id * stride4;
            let mut s = 0.0f32;
            for p in 0..full_pairs {
                let byte = index.codes4[code_off + p];
                let lo = (byte & 0x0F) as usize;
                let hi = (byte >> 4) as usize;
                s += lut4[(2*p) * K4 + lo] + lut4[(2*p + 1) * K4 + hi];
            }
            if has_odd {
                let byte = index.codes4[code_off + full_pairs];
                let lo = (byte & 0x0F) as usize;
                s += lut4[(2*full_pairs) * K4 + lo];
            }
            dists.push(ScanResult { id: id as u32, dist: s });
        }
        // Partition so that dists[..t] holds the T smallest (unordered).
        if t < dists.len() {
            dists.select_nth_unstable_by(t, |a, b| a.dist.partial_cmp(&b.dist).unwrap());
            dists.truncate(t);
        }
        let stage1 = dists;

        // Stage 2: 8-bit refine on survivors.
        let lut8 = index.lut8(query);
        let mut stage2: Vec<ScanResult> = Vec::with_capacity(k + 1);
        for cand in &stage1 {
            let id = cand.id as usize;
            let code_off = id * m;
            let mut s = 0.0f32;
            for j in 0..m {
                let c = index.codes8[code_off + j] as usize;
                s += lut8[j * K8 + c];
            }
            push_topk(&mut stage2, k, ScanResult { id: cand.id, dist: s });
        }
        finalise(stage2, out);
    }
}

/// Push into a bounded top-k structure. For small `k` (≤ ~32) linear insertion
/// on a sorted Vec is cache-friendliest; for large `k` (the Stage-1 survivor
/// set) we use a proper binary max-heap keyed by `-dist` semantics via a
/// hand-rolled sift so we don't pull in the std BinaryHeap wrapper each call.
#[inline]
fn push_topk(heap: &mut Vec<ScanResult>, k: usize, item: ScanResult) {
    if k <= 32 {
        if heap.len() < k {
            let pos = heap.iter().position(|x| x.dist > item.dist).unwrap_or(heap.len());
            heap.insert(pos, item);
        } else if item.dist < heap[k - 1].dist {
            heap.pop();
            let pos = heap.iter().position(|x| x.dist > item.dist).unwrap_or(heap.len());
            heap.insert(pos, item);
        }
    } else {
        // Max-heap by dist: root = largest dist. Push if smaller than root.
        if heap.len() < k {
            heap.push(item);
            let last = heap.len() - 1;
            sift_up(heap, last);
        } else if item.dist < heap[0].dist {
            heap[0] = item;
            sift_down(heap, 0);
        }
    }
}

fn sift_up(h: &mut [ScanResult], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if h[i].dist > h[parent].dist {
            h.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

fn sift_down(h: &mut [ScanResult], mut i: usize) {
    let n = h.len();
    loop {
        let l = 2*i + 1;
        let r = 2*i + 2;
        let mut largest = i;
        if l < n && h[l].dist > h[largest].dist { largest = l; }
        if r < n && h[r].dist > h[largest].dist { largest = r; }
        if largest == i { break; }
        h.swap(i, largest);
        i = largest;
    }
}

fn finalise(heap: Vec<ScanResult>, out: &mut Vec<ScanResult>) {
    out.clear();
    out.extend(heap);
}
