//! PQ Prefix Pruning: partial-sum early abandon for 4-bit FastScan ADC.
//!
//! Standard FastScan accumulates all M partial sums (one per PQ segment) before
//! producing a code's final distance. For top-k search we hold a pivot = current
//! k-th best distance. If a code's partial sum after M' < M segments already
//! exceeds the pivot, remaining segments add non-negative distance and the code
//! cannot enter the top-k. We can abandon it.
//!
//! Two refinements make this real on SIMD paths:
//!   * variance-ordered segments: visit high-variance (high per-segment distance
//!     spread) segments first so the running sum climbs faster, raising the
//!     fraction of codes that trip the pivot early.
//!   * SIMD-group abandon: PQ codes are processed in groups of G = 16 or 32.
//!     Instead of per-lane abandon (which breaks SIMD), we abandon the entire
//!     group once every lane in the group is above the pivot.
//!
//! The crate exposes a `Scanner` trait and three implementations: Baseline,
//! PrefixPrune, and VarOrderedPrune. All operate on 4-bit PQ codes in the
//! canonical FastScan layout (M segments × N codes stored as nibbles).

use std::cmp::Ordering;

pub mod codec;
pub mod lut;
pub mod scan;
pub mod bench_util;

pub use codec::{PqCodes, PqParams};
pub use lut::QueryLut;
pub use scan::{Baseline, PrefixPrune, Scanner, ScannerStats, VarOrderedPrune};

/// A top-k hit: (code_index, distance).
#[derive(Copy, Clone, Debug)]
pub struct Hit {
    pub id: u32,
    pub dist: u16,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}

/// Simple max-heap of fixed capacity k. Pops the worst (largest) when full.
#[derive(Clone, Debug)]
pub struct TopK {
    k: usize,
    buf: Vec<Hit>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, buf: Vec::with_capacity(k + 1) }
    }
    pub fn pivot(&self) -> u16 {
        if self.buf.len() < self.k {
            u16::MAX
        } else {
            self.buf[0].dist
        }
    }
    pub fn push(&mut self, hit: Hit) {
        if self.buf.len() < self.k {
            self.buf.push(hit);
            self.sift_up(self.buf.len() - 1);
        } else if hit.dist < self.buf[0].dist {
            self.buf[0] = hit;
            self.sift_down(0);
        }
    }
    pub fn into_sorted(mut self) -> Vec<Hit> {
        self.buf.sort();
        self.buf
    }
    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.buf[i].dist > self.buf[p].dist {
                self.buf.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }
    fn sift_down(&mut self, mut i: usize) {
        let n = self.buf.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut m = i;
            if l < n && self.buf[l].dist > self.buf[m].dist {
                m = l;
            }
            if r < n && self.buf[r].dist > self.buf[m].dist {
                m = r;
            }
            if m == i {
                break;
            }
            self.buf.swap(i, m);
            i = m;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_pivot_behaviour() {
        let mut h = TopK::new(3);
        assert_eq!(h.pivot(), u16::MAX);
        h.push(Hit { id: 0, dist: 10 });
        h.push(Hit { id: 1, dist: 5 });
        h.push(Hit { id: 2, dist: 20 });
        assert_eq!(h.pivot(), 20);
        h.push(Hit { id: 3, dist: 15 });
        assert_eq!(h.pivot(), 15);
        h.push(Hit { id: 4, dist: 1 });
        assert_eq!(h.pivot(), 10);
        let out = h.into_sorted();
        assert_eq!(out[0].dist, 1);
        assert_eq!(out[1].dist, 5);
        assert_eq!(out[2].dist, 10);
    }
}
