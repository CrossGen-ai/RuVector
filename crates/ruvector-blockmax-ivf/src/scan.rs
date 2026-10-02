//! Three scan backends over an IVF-PQ index.
//!
//! * `SequentialScanner` — baseline: compute full ADC for every code.
//! * `BlockMaxScanner` — per-query block lower bound using the elementwise
//!   min code precomputed at build time. Looks up `tab_j[lb_codes_j]` for each
//!   subspace; sum is a valid lower bound (every code in the block has
//!   `code_j >= lb_j` only in index value — NOT in distance). Because PQ
//!   codes are unordered k-means ids, the stored min-code lower bound is a
//!   *heuristic* lower bound: fast but not provably sound. We measure recall
//!   impact.
//! * `BlockMaxTightScanner` — provably sound: per-query, per-block, take
//!   `min_k_in_block tab_j[code_j_k]` across the block for each subspace;
//!   sum is the exact minimum achievable ADC in that block. One extra pass
//!   per block but still much cheaper than the full K-per-block sum we would
//!   do otherwise.

use crate::ivfpq::IvfPq;
use crate::topk::{Hit, TopK};

#[derive(Default, Debug, Clone, Copy)]
pub struct ScanStats {
    pub blocks_total: u64,
    pub blocks_skipped: u64,
    pub codes_evaluated: u64,
}

pub trait Scanner {
    fn search(&self, index: &IvfPq, query: &[f32], k: usize, nprobe: usize) -> (Vec<Hit>, ScanStats);
}

pub struct SequentialScanner;

impl Scanner for SequentialScanner {
    fn search(&self, index: &IvfPq, query: &[f32], k: usize, nprobe: usize) -> (Vec<Hit>, ScanStats) {
        let mut heap = TopK::new(k);
        let mut stats = ScanStats::default();
        for c in index.probe(query, nprobe) {
            let tab = index.build_lut(query, c);
            let list = &index.lists[c];
            for b in &list.blocks {
                stats.blocks_total += 1;
                let bs = b.ids.len();
                for e in 0..bs {
                    let mut s = 0f32;
                    for j in 0..index.m {
                        s += tab[j * index.ksub + b.codes[e * index.m + j] as usize];
                    }
                    stats.codes_evaluated += 1;
                    heap.push(Hit { id: b.ids[e], dist: s });
                }
            }
        }
        (heap.into_sorted_vec(), stats)
    }
}

pub struct BlockMaxScanner;

impl Scanner for BlockMaxScanner {
    fn search(&self, index: &IvfPq, query: &[f32], k: usize, nprobe: usize) -> (Vec<Hit>, ScanStats) {
        let mut heap = TopK::new(k);
        let mut stats = ScanStats::default();
        for c in index.probe(query, nprobe) {
            let tab = index.build_lut(query, c);
            let list = &index.lists[c];
            for b in &list.blocks {
                stats.blocks_total += 1;
                // Heuristic lower bound from stored elementwise-min codes.
                let mut lb = 0f32;
                for j in 0..index.m {
                    lb += tab[j * index.ksub + b.lb_codes[j] as usize];
                }
                if lb >= heap.threshold() {
                    stats.blocks_skipped += 1;
                    continue;
                }
                let bs = b.ids.len();
                for e in 0..bs {
                    let mut s = 0f32;
                    for j in 0..index.m {
                        s += tab[j * index.ksub + b.codes[e * index.m + j] as usize];
                    }
                    stats.codes_evaluated += 1;
                    heap.push(Hit { id: b.ids[e], dist: s });
                }
            }
        }
        (heap.into_sorted_vec(), stats)
    }
}

pub struct BlockMaxTightScanner;

/// Tight lower bound using per-subspace code-presence bitmasks precomputed at
/// build time. Per block the lb cost is `sum_j popcount(mask_j)` LUT lookups,
/// which is typically ~`unique_codes_per_subspace` ≪ `block_size`. This is the
/// scanner we recommend in practice.
pub struct BlockMaxBitmaskScanner;

impl Scanner for BlockMaxBitmaskScanner {
    fn search(&self, index: &IvfPq, query: &[f32], k: usize, nprobe: usize) -> (Vec<Hit>, ScanStats) {
        let mut heap = TopK::new(k);
        let mut stats = ScanStats::default();
        for c in index.probe(query, nprobe) {
            let tab = index.build_lut(query, c);
            let list = &index.lists[c];
            for b in &list.blocks {
                stats.blocks_total += 1;
                let mut lb = 0f32;
                for j in 0..index.m {
                    let base = j * index.ksub;
                    let mut mn = f32::INFINITY;
                    for w in 0..4 {
                        let mut word = b.code_masks[j * 4 + w];
                        let word_base = w * 64;
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            let v = tab[base + word_base + bit];
                            if v < mn { mn = v; }
                            word &= word - 1;
                        }
                    }
                    lb += mn;
                }
                if lb >= heap.threshold() {
                    stats.blocks_skipped += 1;
                    continue;
                }
                let bs = b.ids.len();
                for e in 0..bs {
                    let mut s = 0f32;
                    for j in 0..index.m {
                        s += tab[j * index.ksub + b.codes[e * index.m + j] as usize];
                    }
                    stats.codes_evaluated += 1;
                    heap.push(Hit { id: b.ids[e], dist: s });
                }
            }
        }
        (heap.into_sorted_vec(), stats)
    }
}

impl Scanner for BlockMaxTightScanner {
    fn search(&self, index: &IvfPq, query: &[f32], k: usize, nprobe: usize) -> (Vec<Hit>, ScanStats) {
        let mut heap = TopK::new(k);
        let mut stats = ScanStats::default();
        for c in index.probe(query, nprobe) {
            let tab = index.build_lut(query, c);
            let list = &index.lists[c];
            for b in &list.blocks {
                stats.blocks_total += 1;
                let bs = b.ids.len();
                // Provably-sound lower bound: per subspace, the min table value
                // over codes appearing in this block.
                let mut lb = 0f32;
                for j in 0..index.m {
                    let mut mn = f32::INFINITY;
                    for e in 0..bs {
                        let v = tab[j * index.ksub + b.codes[e * index.m + j] as usize];
                        if v < mn { mn = v; }
                    }
                    lb += mn;
                }
                if lb >= heap.threshold() {
                    stats.blocks_skipped += 1;
                    continue;
                }
                for e in 0..bs {
                    let mut s = 0f32;
                    for j in 0..index.m {
                        s += tab[j * index.ksub + b.codes[e * index.m + j] as usize];
                    }
                    stats.codes_evaluated += 1;
                    heap.push(Hit { id: b.ids[e], dist: s });
                }
            }
        }
        (heap.into_sorted_vec(), stats)
    }
}
