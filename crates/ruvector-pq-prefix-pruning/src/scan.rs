//! Three ADC scanners sharing a `Scanner` trait.
//!
//! * `Baseline`: process every segment for every code, update topk at the end.
//! * `PrefixPrune`: natural segment order, group of 32 lanes, abandon a group
//!   once every lane's partial sum exceeds `pivot + remaining_floor`.
//! * `VarOrderedPrune`: same but visits segments in descending variance order.
//!
//! `remaining_floor` is the sum of segment minima for the yet-unvisited
//! segments. Adding it to the partial sum tightens the lower bound on the
//! final distance — a code can only enter top-k if partial_sum + floor < pivot.
//! Without it, pruning fires only when partial_sum alone exceeds pivot, which
//! leaves many codes alive even when they cannot win. This mirrors the
//! "lower-bound ADC" trick from the FastScan / RaBitQ literature.

use crate::codec::PqCodes;
use crate::lut::QueryLut;
use crate::{Hit, TopK};

pub const LANE: usize = 32;

#[derive(Default, Debug, Clone, Copy)]
pub struct ScannerStats {
    /// Number of (code × segment) distance lookups actually performed.
    pub seg_lookups: u64,
    /// Number of SIMD-sized groups fully pruned.
    pub groups_pruned: u64,
    /// Number of SIMD-sized groups scanned.
    pub groups_total: u64,
}

pub trait Scanner {
    fn name(&self) -> &'static str;
    fn scan(&mut self, codes: &PqCodes, lut: &QueryLut, k: usize) -> (Vec<Hit>, ScannerStats);
}

/// Baseline: visit every segment for every code; no pruning.
pub struct Baseline;

impl Scanner for Baseline {
    fn name(&self) -> &'static str { "baseline" }
    fn scan(&mut self, codes: &PqCodes, lut: &QueryLut, k: usize) -> (Vec<Hit>, ScannerStats) {
        let n = codes.n;
        let m = codes.params.m;
        let mut heap = TopK::new(k);
        let mut stats = ScannerStats::default();
        for g0 in (0..n).step_by(LANE) {
            let gN = (g0 + LANE).min(n);
            stats.groups_total += 1;
            let mut lane_sum = [0u16; LANE];
            for seg in 0..m {
                for id in g0..gN {
                    let c = codes.code(seg, id);
                    lane_sum[id - g0] += lut.get(seg, c) as u16;
                }
                stats.seg_lookups += (gN - g0) as u64;
            }
            for id in g0..gN {
                heap.push(Hit { id: id as u32, dist: lane_sum[id - g0] });
            }
        }
        (heap.into_sorted(), stats)
    }
}

/// Natural-order prefix pruning.
pub struct PrefixPrune;

impl Scanner for PrefixPrune {
    fn name(&self) -> &'static str { "prefix_prune" }
    fn scan(&mut self, codes: &PqCodes, lut: &QueryLut, k: usize) -> (Vec<Hit>, ScannerStats) {
        scan_pruned(codes, lut, k, (0..codes.params.m).collect(), "prefix_prune")
    }
}

/// Variance-ordered prefix pruning.
pub struct VarOrderedPrune;

impl Scanner for VarOrderedPrune {
    fn name(&self) -> &'static str { "var_ordered" }
    fn scan(&mut self, codes: &PqCodes, lut: &QueryLut, k: usize) -> (Vec<Hit>, ScannerStats) {
        let order = lut.order_by_variance();
        scan_pruned(codes, lut, k, order, "var_ordered")
    }
}

fn scan_pruned(
    codes: &PqCodes,
    lut: &QueryLut,
    k: usize,
    seg_order: Vec<usize>,
    _label: &str,
) -> (Vec<Hit>, ScannerStats) {
    let n = codes.n;
    let m = codes.params.m;
    assert_eq!(seg_order.len(), m);

    // Precompute suffix floor: floor_after[i] = sum of seg_min for seg_order[i..m].
    let mut floor_after = vec![0u16; m + 1];
    for i in (0..m).rev() {
        floor_after[i] = floor_after[i + 1] + lut.seg_min[seg_order[i]] as u16;
    }

    let mut heap = TopK::new(k);
    let mut stats = ScannerStats::default();

    for g0 in (0..n).step_by(LANE) {
        let gN = (g0 + LANE).min(n);
        let lanes = gN - g0;
        stats.groups_total += 1;
        let mut lane_sum = [0u16; LANE];
        let mut alive = [true; LANE];
        let mut alive_count = lanes;

        for (step, &seg) in seg_order.iter().enumerate() {
            if alive_count == 0 {
                break;
            }
            let pivot = heap.pivot();
            let floor = floor_after[step + 1];
            for id in g0..gN {
                let lane = id - g0;
                if !alive[lane] {
                    continue;
                }
                let c = codes.code(seg, id);
                lane_sum[lane] = lane_sum[lane].saturating_add(lut.get(seg, c) as u16);
                stats.seg_lookups += 1;
                // Can this lane still win? sum + remaining_floor must be < pivot.
                if lane_sum[lane].saturating_add(floor) >= pivot {
                    alive[lane] = false;
                    alive_count -= 1;
                }
            }
        }

        if alive_count == 0 {
            // Entire group pruned before completing all segments.
            stats.groups_pruned += 1;
        }

        // Codes still alive (or completed) are candidates. For completed codes
        // their partial sum IS the final distance. For pruned codes we skip —
        // they cannot enter top-k by construction.
        for id in g0..gN {
            let lane = id - g0;
            if alive[lane] {
                heap.push(Hit { id: id as u32, dist: lane_sum[lane] });
            }
        }
    }

    (heap.into_sorted(), stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_util::make_corpus;

    #[test]
    fn prune_matches_baseline_on_random_data() {
        let (vecs, _queries, book, codes) = make_corpus(512, 4, 64, 8, 42);
        let q = vec![0.1f32; book.params.d()];
        let lut = QueryLut::build(&book, &q);
        let k = 10;

        let (base, _) = Baseline.scan(&codes, &lut, k);
        let (prune, s1) = PrefixPrune.scan(&codes, &lut, k);
        let (var, s2) = VarOrderedPrune.scan(&codes, &lut, k);

        // Pruned scanners must return exactly the same top-k (same ids and dists)
        // as baseline — pruning is lossless when seg_min is correct.
        let base_ids: Vec<u32> = base.iter().map(|h| h.id).collect();
        let prune_ids: Vec<u32> = prune.iter().map(|h| h.id).collect();
        let var_ids: Vec<u32> = var.iter().map(|h| h.id).collect();
        assert_eq!(prune_ids, base_ids);
        assert_eq!(var_ids, base_ids);
        for (a, b) in base.iter().zip(prune.iter()) {
            assert_eq!(a.dist, b.dist);
        }
        // Pruning should drop work vs baseline on this corpus.
        let baseline_lookups = codes.n as u64 * codes.params.m as u64;
        assert!(s1.seg_lookups <= baseline_lookups);
        assert!(s2.seg_lookups <= baseline_lookups);
    }
}
