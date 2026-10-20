//! Hamming-weight pre-bucketing for binary ANN search.
//!
//! Binary codes (D bits) are pre-partitioned by their popcount w = pop(x) in
//! [0..D]. For a query q with popcount w_q, the Hamming distance obeys
//!   H(x, q) >= |w(x) - w(q)|
//! so for a k-NN with current worst-known distance tau we may prune any bucket
//! whose weight is further than tau from w_q. This gives a cheap pre-scan
//! triangle bound that composes with any inner Hamming scanner.
//!
//! The crate is deliberately standalone: it depends only on std and ships a
//! trait-based `BinaryScanner` so backends (plain popcount, bit-sliced, SIMD)
//! can be swapped. We bundle three real scanners used in benchmarks:
//!   * `FullScan`       — baseline: scan every code
//!   * `WeightedScan`   — Hamming-weight pre-bucketed scan (this work)
//!   * `WeightedBitSlice` — pre-bucket + bit-sliced popcount within each bucket
//!
//! All operations are deterministic for a fixed seed and bit-width.

#![forbid(unsafe_code)]

use std::collections::BinaryHeap;

/// D in units of u64 words. 2 means 128 bits; 4 means 256 bits.
pub type Code<const W: usize> = [u64; W];

#[inline]
pub fn popcount<const W: usize>(c: &Code<W>) -> u32 {
    let mut s = 0u32;
    for w in c.iter() {
        s += w.count_ones();
    }
    s
}

#[inline]
pub fn hamming<const W: usize>(a: &Code<W>, b: &Code<W>) -> u32 {
    let mut s = 0u32;
    for i in 0..W {
        s += (a[i] ^ b[i]).count_ones();
    }
    s
}

/// Project a float vector to a binary code via sign of (dot product with a
/// deterministic sign-random matrix). This is SimHash with +/- entries; it is
/// cheap, allocation-free given a prepared sign table, and asymptotically
/// equivalent to a Gaussian projection on the unit sphere.
pub fn simhash_encode<const W: usize>(x: &[f32], signs: &SignTable) -> Code<W> {
    assert_eq!(signs.dim, x.len());
    let bits = W * 64;
    assert_eq!(signs.bits, bits);
    let mut code = [0u64; W];
    for b in 0..bits {
        let row = &signs.rows[b];
        let mut acc = 0.0f32;
        // row stores +1/-1 as packed bits: 1 means +1, 0 means -1.
        for (i, &xi) in x.iter().enumerate() {
            let s = ((row[i / 64] >> (i % 64)) & 1) as i32;
            let sign = (s << 1) - 1; // 0->-1, 1->+1
            acc += (sign as f32) * xi;
        }
        if acc >= 0.0 {
            code[b / 64] |= 1u64 << (b % 64);
        }
    }
    code
}

pub struct SignTable {
    pub dim: usize,
    pub bits: usize,
    pub rows: Vec<Vec<u64>>, // one row per bit; each row is dim bits packed
}

impl SignTable {
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        let words = dim.div_ceil(64);
        let mut rows = Vec::with_capacity(bits);
        let mut rng = SplitMix64::new(seed);
        for _ in 0..bits {
            let mut row = vec![0u64; words];
            for w in row.iter_mut() {
                *w = rng.next_u64();
            }
            rows.push(row);
        }
        Self { dim, bits, rows }
    }
}

/// Deterministic 64-bit RNG (SplitMix64) so benchmarks are reproducible.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn gauss(&mut self) -> f32 {
        // Box-Muller
        let u1 = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        let u2 = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        let r = (-2.0 * (u1.max(1e-12)).ln()).sqrt();
        let t = 2.0 * std::f32::consts::PI * u2;
        r * t.cos()
    }
}

pub trait BinaryScanner<const W: usize> {
    /// Return top-k nearest ids with their Hamming distance. Caller provides
    /// a scratch heap-friendly bound; implementations MUST return k pairs
    /// unless the dataset is smaller.
    fn knn(&self, query: &Code<W>, k: usize) -> Vec<(u32, u32)>;
    fn name(&self) -> &'static str;
    /// Count of candidate codes actually visited by the last knn call.
    fn visited(&self) -> u64;
    fn reset_counters(&self);
}

/// Baseline: visit every code.
pub struct FullScan<const W: usize> {
    pub codes: Vec<Code<W>>,
    visited: std::cell::Cell<u64>,
}

impl<const W: usize> FullScan<W> {
    pub fn new(codes: Vec<Code<W>>) -> Self {
        Self { codes, visited: std::cell::Cell::new(0) }
    }
}

impl<const W: usize> BinaryScanner<W> for FullScan<W> {
    fn knn(&self, query: &Code<W>, k: usize) -> Vec<(u32, u32)> {
        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
        for (i, c) in self.codes.iter().enumerate() {
            let d = hamming(c, query);
            if heap.len() < k {
                heap.push((d, i as u32));
            } else if d < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((d, i as u32));
            }
        }
        self.visited.set(self.codes.len() as u64);
        let mut out: Vec<(u32, u32)> = heap.into_iter().map(|(d, i)| (i, d)).collect();
        out.sort_by_key(|(_, d)| *d);
        out
    }
    fn name(&self) -> &'static str { "full" }
    fn visited(&self) -> u64 { self.visited.get() }
    fn reset_counters(&self) { self.visited.set(0); }
}

/// Hamming-weight pre-bucketed scan (this work). Codes are partitioned by
/// their popcount w in 0..=D. At query time we expand buckets outward from
/// the query's own weight w_q, maintaining the current k-th-best Hamming tau,
/// and stop as soon as |w - w_q| > tau. The expansion is a tight proof of
/// optimality: any unseen bucket has every member at Hamming distance >=
/// |w - w_q|, which already exceeds the current tau.
pub struct WeightedScan<const W: usize> {
    /// buckets[w] = ids of codes whose popcount equals w
    pub buckets: Vec<Vec<u32>>,
    pub codes: Vec<Code<W>>,
    pub dim_bits: u32,
    visited: std::cell::Cell<u64>,
}

impl<const W: usize> WeightedScan<W> {
    pub fn build(codes: Vec<Code<W>>) -> Self {
        let bits = (W as u32) * 64;
        let mut buckets: Vec<Vec<u32>> = (0..=bits).map(|_| Vec::new()).collect();
        for (i, c) in codes.iter().enumerate() {
            let w = popcount(c) as usize;
            buckets[w].push(i as u32);
        }
        Self { buckets, codes, dim_bits: bits, visited: std::cell::Cell::new(0) }
    }
}

impl<const W: usize> BinaryScanner<W> for WeightedScan<W> {
    fn knn(&self, query: &Code<W>, k: usize) -> Vec<(u32, u32)> {
        let wq = popcount(query) as i32;
        let bits = self.dim_bits as i32;
        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
        let mut visited = 0u64;

        // Expand outward in weight: wq, wq-1, wq+1, wq-2, wq+2, ...
        // Stop when |delta| exceeds current tau.
        for delta in 0..=bits {
            let tau = if heap.len() == k { heap.peek().unwrap().0 } else { u32::MAX };
            if (delta as u32) > tau { break; }
            for sign in [-1i32, 1i32].iter() {
                if delta == 0 && *sign == 1 { continue; } // avoid double-visit at wq
                let w = if delta == 0 { wq } else if *sign == -1 { wq - delta } else { wq + delta };
                if w < 0 || w > bits { continue; }
                for &id in &self.buckets[w as usize] {
                    visited += 1;
                    let d = hamming(&self.codes[id as usize], query);
                    if heap.len() < k {
                        heap.push((d, id));
                    } else if d < heap.peek().unwrap().0 {
                        heap.pop();
                        heap.push((d, id));
                    }
                }
            }
        }
        self.visited.set(visited);
        let mut out: Vec<(u32, u32)> = heap.into_iter().map(|(d, i)| (i, d)).collect();
        out.sort_by_key(|(_, d)| *d);
        out
    }
    fn name(&self) -> &'static str { "weight-bucket" }
    fn visited(&self) -> u64 { self.visited.get() }
    fn reset_counters(&self) { self.visited.set(0); }
}

/// Pre-bucket + bit-sliced popcount: within each visited bucket, compute
/// Hamming distances in a tight SIMD-friendly loop over u64 words rather
/// than a per-code call. For small W this is a micro-optimization but it
/// shows the pre-bucket layer composes cleanly with any inner scanner.
pub struct WeightedBitSlice<const W: usize> {
    pub buckets: Vec<Vec<u32>>,
    pub codes_flat: Vec<u64>,       // W * N packed row-major: code i at [i*W..i*W+W]
    pub dim_bits: u32,
    visited: std::cell::Cell<u64>,
    n: usize,
}

impl<const W: usize> WeightedBitSlice<W> {
    pub fn build(codes: Vec<Code<W>>) -> Self {
        let bits = (W as u32) * 64;
        let n = codes.len();
        let mut buckets: Vec<Vec<u32>> = (0..=bits).map(|_| Vec::new()).collect();
        let mut flat = Vec::with_capacity(n * W);
        for (i, c) in codes.iter().enumerate() {
            let w = popcount(c) as usize;
            buckets[w].push(i as u32);
            flat.extend_from_slice(c);
        }
        Self { buckets, codes_flat: flat, dim_bits: bits, visited: std::cell::Cell::new(0), n }
    }
    #[inline]
    fn code_dist(&self, i: usize, query: &Code<W>) -> u32 {
        let base = i * W;
        let mut s = 0u32;
        for k in 0..W {
            s += (self.codes_flat[base + k] ^ query[k]).count_ones();
        }
        s
    }
}

impl<const W: usize> BinaryScanner<W> for WeightedBitSlice<W> {
    fn knn(&self, query: &Code<W>, k: usize) -> Vec<(u32, u32)> {
        let wq = popcount(query) as i32;
        let bits = self.dim_bits as i32;
        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
        let mut visited = 0u64;
        for delta in 0..=bits {
            let tau = if heap.len() == k { heap.peek().unwrap().0 } else { u32::MAX };
            if (delta as u32) > tau { break; }
            for sign in [-1i32, 1i32].iter() {
                if delta == 0 && *sign == 1 { continue; }
                let w = if delta == 0 { wq } else if *sign == -1 { wq - delta } else { wq + delta };
                if w < 0 || w > bits { continue; }
                for &id in &self.buckets[w as usize] {
                    visited += 1;
                    let d = self.code_dist(id as usize, query);
                    if heap.len() < k {
                        heap.push((d, id));
                    } else if d < heap.peek().unwrap().0 {
                        heap.pop();
                        heap.push((d, id));
                    }
                }
            }
        }
        self.visited.set(visited);
        let mut out: Vec<(u32, u32)> = heap.into_iter().map(|(d, i)| (i, d)).collect();
        out.sort_by_key(|(_, d)| *d);
        out
    }
    fn name(&self) -> &'static str { "weight-bucket+bitslice" }
    fn visited(&self) -> u64 { self.visited.get() }
    fn reset_counters(&self) { self.visited.set(0); }
}

/// Estimated bytes for a bucket index: just the id vectors (u32).
pub fn bucket_bytes<const W: usize>(buckets: &[Vec<u32>]) -> usize {
    buckets.iter().map(|b| b.len() * std::mem::size_of::<u32>() + std::mem::size_of::<Vec<u32>>()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_codes(n: usize, seed: u64) -> Vec<Code<2>> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| [rng.next_u64(), rng.next_u64()]).collect()
    }

    #[test]
    fn popcount_hamming_consistency() {
        let a: Code<2> = [0u64, 0u64];
        let b: Code<2> = [u64::MAX, u64::MAX];
        assert_eq!(popcount(&a), 0);
        assert_eq!(popcount(&b), 128);
        assert_eq!(hamming(&a, &b), 128);
        assert_eq!(hamming(&a, &a), 0);
    }

    #[test]
    fn weight_bucket_matches_full_scan() {
        let codes = mk_codes(2_000, 42);
        let full = FullScan::new(codes.clone());
        let wb = WeightedScan::build(codes.clone());
        let wbs = WeightedBitSlice::build(codes.clone());
        let mut rng = SplitMix64::new(123);
        for _ in 0..32 {
            let q: Code<2> = [rng.next_u64(), rng.next_u64()];
            let g = full.knn(&q, 10);
            let a = wb.knn(&q, 10);
            let b = wbs.knn(&q, 10);
            // The ground truth is defined by Hamming distances. Ties on the
            // k-th distance are allowed, so compare the sorted distances only.
            let gd: Vec<u32> = g.iter().map(|(_, d)| *d).collect();
            let ad: Vec<u32> = a.iter().map(|(_, d)| *d).collect();
            let bd: Vec<u32> = b.iter().map(|(_, d)| *d).collect();
            assert_eq!(gd, ad, "weight-bucket distances must match full scan");
            assert_eq!(gd, bd, "bit-sliced variant must match full scan");
            assert!(wb.visited() <= full.visited(), "bucket scan must not visit more than full");
        }
    }

    #[test]
    fn simhash_round_trip() {
        let signs = SignTable::new(32, 128, 7);
        let x: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let c = simhash_encode::<2>(&x, &signs);
        // Same vector encodes to same code; negation of vector flips every bit.
        let c2 = simhash_encode::<2>(&x, &signs);
        assert_eq!(c, c2);
        let neg: Vec<f32> = x.iter().map(|v| -v).collect();
        let cn = simhash_encode::<2>(&neg, &signs);
        assert_eq!(hamming(&c, &cn), 128, "sign flip must flip every bit");
    }

    #[test]
    fn pruning_reduces_visits() {
        // Bimodal popcount distribution: half the codes have ~8 ones, half have
        // ~56 ones. A query drawn from either mode has small tau against its
        // own mode, which excludes every bucket of the opposite mode.
        let mut rng = SplitMix64::new(99);
        let mut mk = |target_pop: usize, rng: &mut SplitMix64| -> Code<2> {
            let mut c = [0u64; 2];
            let mut set = 0usize;
            while set < target_pop {
                let b = (rng.next_u64() % 128) as usize;
                let mask = 1u64 << (b % 64);
                if c[b / 64] & mask == 0 { c[b / 64] |= mask; set += 1; }
            }
            c
        };
        let mut codes = Vec::with_capacity(5_000);
        for _ in 0..2_500 { codes.push(mk(8, &mut rng)); }
        for _ in 0..2_500 { codes.push(mk(56, &mut rng)); }
        let full = FullScan::new(codes.clone());
        let wb = WeightedScan::build(codes.clone());
        let mut rng2 = SplitMix64::new(7);
        let mut saved = 0u64;
        let n_q = 32;
        for _ in 0..n_q {
            let q = mk(8, &mut rng2);
            let _ = full.knn(&q, 10);
            let _ = wb.knn(&q, 10);
            saved += full.visited().saturating_sub(wb.visited());
        }
        assert!(saved > 0, "weight pruning should save visits on bimodal weights");
    }
}
