//! PDX with dimension-order pruning.
//!
//! After walking `dim_step` dimensions of a block, compare each lane's
//! running partial L2² against the current top-k threshold. Lanes whose
//! partial already exceeds the threshold are marked inactive; subsequent
//! dimension groups skip them entirely.
//!
//! The dimension permutation is chosen at build time. Variance-descending
//! is the paper's recommendation: high-variance dimensions contribute more
//! to distance early, driving lanes across the threshold sooner.

use crate::block::{HorizontalStore, PdxStore, VectorStore, BLOCK_LEN};
use crate::scan::TopK;

/// Dimension ordering used when building a `PdxStore`.
#[derive(Copy, Clone, Debug)]
pub enum DimensionOrder {
    /// Identity (original dim ids 0..d).
    Identity,
    /// Descending sample variance across the dataset. Best pruning.
    VarianceDesc,
}

impl DimensionOrder {
    pub fn permutation(self, store: &HorizontalStore) -> Vec<u32> {
        let d = store.dim();
        match self {
            DimensionOrder::Identity => (0..d as u32).collect(),
            DimensionOrder::VarianceDesc => {
                let n = store.len().max(1);
                let mut mean = vec![0f64; d];
                for id in 0..store.len() {
                    for (j, &x) in store.row(id).iter().enumerate() {
                        mean[j] += x as f64;
                    }
                }
                for m in &mut mean { *m /= n as f64; }
                let mut var = vec![0f64; d];
                for id in 0..store.len() {
                    for (j, &x) in store.row(id).iter().enumerate() {
                        let diff = x as f64 - mean[j];
                        var[j] += diff * diff;
                    }
                }
                let mut idx: Vec<u32> = (0..d as u32).collect();
                idx.sort_by(|&a, &b| var[b as usize].partial_cmp(&var[a as usize]).unwrap_or(std::cmp::Ordering::Equal));
                idx
            }
        }
    }
}

/// Aggregate stats for one pruned scan: how many (lane, dim) cells were
/// visited vs. the horizontal baseline `n * d`.
#[derive(Copy, Clone, Debug, Default)]
pub struct PruneStats {
    pub cells_visited: u64,
    pub cells_total: u64,
    pub blocks_early_exit: u64,
    pub blocks_total: u64,
}

impl PruneStats {
    pub fn work_ratio(&self) -> f64 {
        if self.cells_total == 0 { return 1.0; }
        self.cells_visited as f64 / self.cells_total as f64
    }
}

/// PDX L2² top-k with per-block pruning after every `dim_step` dims.
///
/// `dim_step` = 8 is a sensible default: enough dimensions to build a real
/// partial distance signal, small enough to prune before too much wasted
/// work on doomed lanes.
pub fn brute_pdx_pruned(
    store: &PdxStore,
    query: &[f32],
    k: usize,
    dim_step: usize,
) -> (Vec<(f32, u32)>, PruneStats) {
    assert_eq!(query.len(), store.dim());
    assert!(dim_step >= 1);

    let perm = store.perm();
    let mut pq = vec![0f32; store.dim()];
    for (kslot, &od) in perm.iter().enumerate() {
        pq[kslot] = query[od as usize];
    }

    let mut topk = TopK::new(k);
    let mut acc = [0f32; BLOCK_LEN];
    let mut active = [true; BLOCK_LEN];
    let d = store.dim();

    let mut stats = PruneStats::default();
    stats.cells_total = (store.len() as u64) * (d as u64);
    stats.blocks_total = store.num_blocks() as u64;

    for b in 0..store.num_blocks() {
        let bl = store.block_size(b);
        acc[..bl].fill(0.0);
        for a in &mut active[..bl] { *a = true; }
        let block = store.block(b);
        let mut done_early = false;

        let mut kslot = 0;
        while kslot < d {
            let end = (kslot + dim_step).min(d);
            for k2 in kslot..end {
                let q = pq[k2];
                let col = &block[k2 * bl..(k2 + 1) * bl];
                for lane in 0..bl {
                    if active[lane] {
                        let diff = col[lane] - q;
                        acc[lane] += diff * diff;
                        stats.cells_visited += 1;
                    }
                }
            }
            kslot = end;

            // Prune against current threshold.
            let thresh = topk.threshold();
            if thresh.is_finite() {
                let mut any_active = false;
                for lane in 0..bl {
                    if active[lane] && acc[lane] > thresh {
                        active[lane] = false;
                    }
                    any_active |= active[lane];
                }
                if !any_active {
                    stats.blocks_early_exit += 1;
                    done_early = true;
                    break;
                }
            }
        }

        // Emit surviving lanes' final distances.
        if !done_early {
            let base = (b * BLOCK_LEN) as u32;
            for lane in 0..bl {
                if active[lane] {
                    topk.offer(acc[lane], base + lane as u32);
                }
            }
        }
    }

    (topk.into_sorted(), stats)
}

/// Branchless variant: full column ops per dim group, prune only at the
/// **whole-block** level. The inner loop stays autovectorizable because it
/// has no per-lane predicate. We test after each `dim_step` if the current
/// **minimum** lane accumulator already exceeds the top-k threshold; when
/// it does, every lane in this block is doomed and we early-exit the whole
/// block. This trades finer-grained lane pruning for SIMD throughput.
pub fn brute_pdx_block_pruned(
    store: &PdxStore,
    query: &[f32],
    k: usize,
    dim_step: usize,
) -> (Vec<(f32, u32)>, PruneStats) {
    assert_eq!(query.len(), store.dim());
    assert!(dim_step >= 1);
    let perm = store.perm();
    let mut pq = vec![0f32; store.dim()];
    for (kslot, &od) in perm.iter().enumerate() {
        pq[kslot] = query[od as usize];
    }
    let mut topk = TopK::new(k);
    let mut acc = [0f32; BLOCK_LEN];
    let d = store.dim();
    let mut stats = PruneStats::default();
    stats.cells_total = (store.len() as u64) * (d as u64);
    stats.blocks_total = store.num_blocks() as u64;

    for b in 0..store.num_blocks() {
        let bl = store.block_size(b);
        acc[..bl].fill(0.0);
        let block = store.block(b);
        let mut done_early = false;
        let mut visited_dims = 0usize;

        let mut kslot = 0;
        while kslot < d {
            let end = (kslot + dim_step).min(d);
            // Branchless tight loop: no per-lane predicate.
            for k2 in kslot..end {
                let q = pq[k2];
                let col = &block[k2 * bl..(k2 + 1) * bl];
                for lane in 0..bl {
                    let diff = col[lane] - q;
                    acc[lane] += diff * diff;
                }
            }
            visited_dims = end;
            kslot = end;
            // Block-level early exit if MIN lane already > threshold.
            let thresh = topk.threshold();
            if thresh.is_finite() {
                let mut min_acc = f32::INFINITY;
                for lane in 0..bl {
                    if acc[lane] < min_acc { min_acc = acc[lane]; }
                }
                if min_acc > thresh {
                    stats.blocks_early_exit += 1;
                    done_early = true;
                    break;
                }
            }
        }
        stats.cells_visited += (bl as u64) * (visited_dims as u64);
        if !done_early {
            let base = (b * BLOCK_LEN) as u32;
            for lane in 0..bl {
                topk.offer(acc[lane], base + lane as u32);
            }
        }
    }
    (topk.into_sorted(), stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::{brute_horizontal, brute_pdx};
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn make_dataset(n: usize, d: usize, seed: u64) -> (HorizontalStore, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
            .collect();
        let query: Vec<f32> = (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        (HorizontalStore::from_iter(d, vecs), query)
    }

    #[test]
    fn pdx_pruned_matches_horizontal() {
        let (h, q) = make_dataset(2000, 64, 42);
        let perm = DimensionOrder::VarianceDesc.permutation(&h);
        let p = PdxStore::from_horizontal(&h, perm);
        let k = 10;
        let base = brute_horizontal(&h, &q, k);
        let plain = brute_pdx(&p, &q, k);
        let (pruned, stats) = brute_pdx_pruned(&p, &q, k, 8);
        assert_eq!(base.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                   pruned.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                   "PDX-pruned top-k must equal horizontal top-k ids");
        assert_eq!(base.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                   plain.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                   "PDX-plain top-k must equal horizontal top-k ids");
        assert!(stats.work_ratio() < 1.0, "pruned scan must skip work; got {:?}", stats);
        for ((_, bi), (_, pi)) in base.iter().zip(pruned.iter()) {
            assert_eq!(bi, pi);
        }
    }

    #[test]
    fn distances_agree_within_epsilon() {
        let (h, q) = make_dataset(500, 48, 7);
        let perm = DimensionOrder::VarianceDesc.permutation(&h);
        let p = PdxStore::from_horizontal(&h, perm);
        let base = brute_horizontal(&h, &q, 5);
        let (pruned, _) = brute_pdx_pruned(&p, &q, 5, 4);
        for ((bd, bi), (pd, pi)) in base.iter().zip(pruned.iter()) {
            assert_eq!(bi, pi);
            assert!((bd - pd).abs() < 1e-3, "dist {bd} vs {pd}");
        }
    }
}
