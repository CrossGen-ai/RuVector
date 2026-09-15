//! IVF (inverted-file) driver that scans each probed list with a chosen
//! backend. Demonstrates PDX's real win: on an IVF list of ~500 candidates,
//! pruning kicks in *fast* because the top-k threshold tightens quickly.

use crate::block::{HorizontalStore, PdxStore, VectorStore};
use crate::pruned::{brute_pdx_pruned, DimensionOrder, PruneStats};
use crate::scan::{brute_horizontal, TopK};

/// Assignments: `lists[c]` = ids of vectors assigned to centroid `c`.
pub struct Assignments { pub lists: Vec<Vec<u32>> }

/// Simple k-means++ style centroid assignment (fixed iterations).
pub fn kmeans_assign(store: &HorizontalStore, n_lists: usize, iters: usize, seed: u64) -> (Vec<Vec<f32>>, Assignments) {
    use rand::prelude::*;
    let d = store.dim();
    let n = store.len();
    let mut rng = StdRng::seed_from_u64(seed);
    // Random init from data.
    let mut centroids: Vec<Vec<f32>> = (0..n_lists).map(|_| store.row(rng.gen_range(0..n)).to_vec()).collect();
    let mut assign = vec![0u32; n];

    for _ in 0..iters {
        // Assign.
        for id in 0..n {
            let v = store.row(id);
            let mut best = (f32::INFINITY, 0u32);
            for (ci, c) in centroids.iter().enumerate() {
                let mut acc = 0f32;
                for j in 0..d { let e = v[j] - c[j]; acc += e * e; }
                if acc < best.0 { best = (acc, ci as u32); }
            }
            assign[id] = best.1;
        }
        // Update.
        let mut sums = vec![vec![0f32; d]; n_lists];
        let mut counts = vec![0u32; n_lists];
        for id in 0..n {
            let c = assign[id] as usize;
            counts[c] += 1;
            let v = store.row(id);
            for j in 0..d { sums[c][j] += v[j]; }
        }
        for (ci, s) in sums.iter().enumerate() {
            if counts[ci] > 0 {
                for j in 0..d { centroids[ci][j] = s[j] / counts[ci] as f32; }
            }
        }
    }

    let mut lists: Vec<Vec<u32>> = (0..n_lists).map(|_| Vec::new()).collect();
    for id in 0..n { lists[assign[id] as usize].push(id as u32); }
    (centroids, Assignments { lists })
}

/// Pick `nprobe` closest centroids to the query.
fn probe(centroids: &[Vec<f32>], query: &[f32], nprobe: usize) -> Vec<u32> {
    let mut d: Vec<(f32, u32)> = centroids.iter().enumerate().map(|(i, c)| {
        let mut acc = 0f32;
        for j in 0..c.len() { let e = c[j] - query[j]; acc += e * e; }
        (acc, i as u32)
    }).collect();
    let split = nprobe.min(d.len().saturating_sub(1));
    d.select_nth_unstable_by(split, |a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(nprobe).map(|(_, i)| i).collect()
}

#[derive(Copy, Clone, Debug, Default)]
pub struct IvfStats {
    pub candidates: u64,
    pub prune: PruneStats,
}

/// IVF driver using horizontal storage per list (baseline).
pub struct IvfHorizontal {
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<HorizontalStore>,
    /// Map from (list, local-id) -> global id.
    pub id_map: Vec<Vec<u32>>,
    pub dim: usize,
}

impl IvfHorizontal {
    pub fn build(store: &HorizontalStore, n_lists: usize, iters: usize, seed: u64) -> Self {
        let (centroids, assign) = kmeans_assign(store, n_lists, iters, seed);
        let mut lists = Vec::with_capacity(n_lists);
        let mut id_map = Vec::with_capacity(n_lists);
        for (ci, ids) in assign.lists.iter().enumerate() {
            let vecs: Vec<Vec<f32>> = ids.iter().map(|&i| store.row(i as usize).to_vec()).collect();
            lists.push(HorizontalStore::from_iter(store.dim(), vecs));
            id_map.push(ids.clone());
            let _ = ci;
        }
        Self { centroids, lists, id_map, dim: store.dim() }
    }

    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> (Vec<(f32, u32)>, IvfStats) {
        let mut heap = TopK::new(k);
        let mut stats = IvfStats::default();
        for c in probe(&self.centroids, query, nprobe) {
            let list = &self.lists[c as usize];
            stats.candidates += list.len() as u64;
            let local_top = brute_horizontal(list, query, k);
            for (d, li) in local_top {
                let gid = self.id_map[c as usize][li as usize];
                heap.offer(d, gid);
            }
        }
        (heap.into_sorted(), stats)
    }
}

/// IVF driver using PDX + variance-order + pruning per list.
pub struct IvfPdxPruned {
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<PdxStore>,
    pub id_map: Vec<Vec<u32>>,
    pub dim: usize,
    pub dim_step: usize,
}

impl IvfPdxPruned {
    pub fn build(store: &HorizontalStore, n_lists: usize, iters: usize, seed: u64, dim_step: usize) -> Self {
        let (centroids, assign) = kmeans_assign(store, n_lists, iters, seed);
        let mut lists = Vec::with_capacity(n_lists);
        let mut id_map = Vec::with_capacity(n_lists);
        for ids in &assign.lists {
            let vecs: Vec<Vec<f32>> = ids.iter().map(|&i| store.row(i as usize).to_vec()).collect();
            let h = HorizontalStore::from_iter(store.dim(), vecs);
            // Per-list variance-descending permutation.
            let perm = DimensionOrder::VarianceDesc.permutation(&h);
            let p = PdxStore::from_horizontal(&h, perm);
            lists.push(p);
            id_map.push(ids.clone());
        }
        Self { centroids, lists, id_map, dim: store.dim(), dim_step }
    }

    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> (Vec<(f32, u32)>, IvfStats) {
        let mut heap = TopK::new(k);
        let mut stats = IvfStats::default();
        for c in probe(&self.centroids, query, nprobe) {
            let list = &self.lists[c as usize];
            stats.candidates += list.len() as u64;
            let (local_top, ls) = brute_pdx_pruned(list, query, k, self.dim_step);
            stats.prune.cells_visited += ls.cells_visited;
            stats.prune.cells_total += ls.cells_total;
            stats.prune.blocks_early_exit += ls.blocks_early_exit;
            stats.prune.blocks_total += ls.blocks_total;
            for (d, li) in local_top {
                let gid = self.id_map[c as usize][li as usize];
                heap.offer(d, gid);
            }
        }
        (heap.into_sorted(), stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::brute_horizontal;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn dataset(n: usize, d: usize, seed: u64) -> (HorizontalStore, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let v: Vec<Vec<f32>> = (0..n).map(|_| (0..d).map(|_| rng.sample::<f32,_>(StandardNormal)).collect()).collect();
        let q: Vec<f32> = (0..d).map(|_| rng.sample::<f32,_>(StandardNormal)).collect();
        (HorizontalStore::from_iter(d, v), q)
    }

    #[test]
    fn ivf_pdx_matches_ivf_horizontal_when_nprobe_full() {
        let (h, q) = dataset(2000, 32, 11);
        let ivf_h = IvfHorizontal::build(&h, 16, 6, 3);
        let ivf_p = IvfPdxPruned::build(&h, 16, 6, 3, 8);
        let (a, _) = ivf_h.search(&q, 10, 16);
        let (b, _) = ivf_p.search(&q, 10, 16);
        let ai: Vec<u32> = a.iter().map(|x| x.1).collect();
        let bi: Vec<u32> = b.iter().map(|x| x.1).collect();
        assert_eq!(ai, bi, "IVF-PDX must equal IVF-horizontal at nprobe=all");
    }

    #[test]
    fn ivf_pdx_recall_at_nprobe_partial() {
        let (h, q) = dataset(4000, 32, 13);
        let ivf_p = IvfPdxPruned::build(&h, 32, 6, 3, 8);
        let base = brute_horizontal(&h, &q, 10);
        let (approx, _) = ivf_p.search(&q, 10, 8);
        let base_ids: std::collections::HashSet<u32> = base.iter().map(|x| x.1).collect();
        let approx_ids: std::collections::HashSet<u32> = approx.iter().map(|x| x.1).collect();
        let recall = base_ids.intersection(&approx_ids).count() as f32 / base_ids.len() as f32;
        assert!(recall >= 0.5, "IVF-PDX recall@10 with nprobe=8/32 must be >= 0.5, got {recall}");
    }
}
