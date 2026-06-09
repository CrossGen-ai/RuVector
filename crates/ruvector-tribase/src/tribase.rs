//! Tribase: triangle-inequality pruning over IVF posting lists.
//!
//! Build time:
//!   * Run k-means to get centroids.
//!   * Assign each point to its nearest centroid.
//!   * Sort each posting list by `d(x, c)`. Store a parallel array of
//!     these centroid distances so query time can binary-search a
//!     window without recomputing them.
//!
//! Query time, for each probed cluster `j`:
//!   * Compute `qd = d(q, c_j)` once.
//!   * The candidate window is `[qd - tau, qd + tau]` where `tau` is
//!     the current k-th best distance in the heap.
//!   * Binary-search for the start and end of that window in the
//!     sorted centroid-distance array.
//!   * Walk only inside the window; for each candidate compute the
//!     real distance and update the heap (which may tighten `tau`).
//!
//! For the very first cluster we have no heap, so `tau = +inf`. After
//! the heap fills, `tau` shrinks and the window narrows, which is
//! where the speedup comes from.

use crate::{kmeans, l2, sq_l2, AnnIndex, Neighbor, SearchStats};
use std::collections::BinaryHeap;

pub struct TribaseIndex {
    pub centroids: Vec<Vec<f32>>,
    /// Per cluster: ids of points sorted by d(x, c) ascending.
    pub posting_ids: Vec<Vec<u32>>,
    /// Per cluster: vectors sorted in the same order as `posting_ids`.
    pub posting_vecs: Vec<Vec<Vec<f32>>>,
    /// Per cluster: d(x, c) values sorted ascending — used for binary
    /// search of the triangle-inequality admissible window.
    pub posting_dists: Vec<Vec<f32>>,
    pub n_probe: usize,
    /// If true, also use the lower-bound check inside the window —
    /// `|qd - xd| > tau` — as a fast skip. This is the actual Tribase
    /// pruning step and is cheap (one subtraction + compare).
    pub use_lower_bound: bool,
}

impl TribaseIndex {
    pub fn build(
        data: Vec<Vec<f32>>,
        n_clusters: usize,
        n_probe: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        let centroids = kmeans(&data, n_clusters, kmeans_iters, seed);
        let mut raw: Vec<Vec<(f32, u32, Vec<f32>)>> = vec![Vec::new(); n_clusters];
        for (i, x) in data.into_iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_l2(&x, c);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            let cd = bd.sqrt();
            raw[best].push((cd, i as u32, x));
        }
        // Sort each posting list by centroid distance.
        let mut posting_ids: Vec<Vec<u32>> = Vec::with_capacity(n_clusters);
        let mut posting_vecs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_clusters);
        let mut posting_dists: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
        for mut bucket in raw {
            bucket.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut ids = Vec::with_capacity(bucket.len());
            let mut vecs = Vec::with_capacity(bucket.len());
            let mut dists = Vec::with_capacity(bucket.len());
            for (cd, id, v) in bucket {
                dists.push(cd);
                ids.push(id);
                vecs.push(v);
            }
            posting_ids.push(ids);
            posting_vecs.push(vecs);
            posting_dists.push(dists);
        }
        Self {
            centroids,
            posting_ids,
            posting_vecs,
            posting_dists,
            n_probe,
            use_lower_bound: true,
        }
    }

    fn probe_clusters(&self, q: &[f32]) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, l2(q, c)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(self.n_probe);
        all
    }

    /// Lower bound on the window inside a sorted `xd` array, i.e.
    /// first index with `xd[i] >= lo`. Standard `partition_point`.
    fn window_lo(xd: &[f32], lo: f32) -> usize {
        xd.partition_point(|&v| v < lo)
    }

    fn window_hi(xd: &[f32], hi: f32) -> usize {
        xd.partition_point(|&v| v <= hi)
    }
}

impl AnnIndex for TribaseIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        self.search_with_stats(q, k).0
    }

    fn search_with_stats(&self, q: &[f32], k: usize) -> (Vec<Neighbor>, SearchStats) {
        let mut stats = SearchStats::default();
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        for (cluster, qd) in self.probe_clusters(q) {
            let ids = &self.posting_ids[cluster];
            let vecs = &self.posting_vecs[cluster];
            let xd = &self.posting_dists[cluster];
            stats.considered += ids.len() as u64;

            // Triangle-inequality window using the current best dist.
            let tau = if heap.len() < k {
                f32::INFINITY
            } else {
                heap.peek().map(|n| n.dist).unwrap_or(f32::INFINITY)
            };
            let lo = if tau.is_finite() { qd - tau } else { f32::NEG_INFINITY };
            let hi = if tau.is_finite() { qd + tau } else { f32::INFINITY };
            let s = Self::window_lo(xd, lo);
            let e = Self::window_hi(xd, hi);

            // Anything outside [s, e) is pruned by the window itself.
            stats.pruned += (ids.len() - (e - s)) as u64;

            for slot in s..e {
                // Tighten tau as the heap fills inside this cluster.
                let cur_tau = if heap.len() < k {
                    f32::INFINITY
                } else {
                    heap.peek().map(|n| n.dist).unwrap_or(f32::INFINITY)
                };
                if self.use_lower_bound && cur_tau.is_finite() {
                    let lb = (qd - xd[slot]).abs();
                    if lb > cur_tau {
                        stats.pruned += 1;
                        continue;
                    }
                }
                let d = l2(q, &vecs[slot]);
                stats.full_dist += 1;
                if heap.len() < k {
                    heap.push(Neighbor { id: ids[slot], dist: d });
                } else if let Some(top) = heap.peek() {
                    if d < top.dist {
                        heap.pop();
                        heap.push(Neighbor { id: ids[slot], dist: d });
                    }
                }
            }
        }
        let mut out: Vec<Neighbor> = heap.into_sorted_vec();
        out.truncate(k);
        (out, stats)
    }

    fn estimated_bytes(&self) -> usize {
        let f32sz = std::mem::size_of::<f32>();
        let dim = self.centroids.first().map(|v| v.len()).unwrap_or(0);
        let cent = self.centroids.len() * dim * f32sz;
        let vecs: usize = self.posting_vecs.iter().map(|p| p.len() * dim * f32sz).sum();
        let ids: usize = self.posting_ids.iter().map(|p| p.len() * 4).sum();
        let dists: usize = self.posting_dists.iter().map(|p| p.len() * f32sz).sum();
        cent + vecs + ids + dists
    }

    fn name(&self) -> &'static str {
        "ivf-tribase"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_clustered, FlatIndex, PlainIvfIndex};

    #[test]
    fn tribase_matches_plain_ivf() {
        let data = make_clustered(1500, 16, 12, 0.08, 23);
        let plain = PlainIvfIndex::build(data.clone(), 12, 12, 25, 9);
        let tri = TribaseIndex::build(data.clone(), 12, 12, 25, 9);
        let q = data[42].clone();
        let a = plain.search(&q, 10);
        let b = tri.search(&q, 10);
        // Same probe set, same data — pruning is correctness-preserving.
        let ids_a: Vec<u32> = a.iter().map(|n| n.id).collect();
        let ids_b: Vec<u32> = b.iter().map(|n| n.id).collect();
        assert_eq!(ids_a, ids_b);
    }

    #[test]
    fn tribase_top1_recovers_self() {
        let data = make_clustered(1000, 24, 10, 0.05, 1);
        let tri = TribaseIndex::build(data.clone(), 10, 10, 25, 1);
        let q = data[777].clone();
        let r = tri.search(&q, 1);
        assert_eq!(r[0].id, 777);
    }

    #[test]
    fn tribase_prunes_strictly_more_than_plain() {
        let data = make_clustered(4_000, 32, 16, 0.04, 5);
        let plain = PlainIvfIndex::build(data.clone(), 16, 6, 25, 7);
        let tri = TribaseIndex::build(data.clone(), 16, 6, 25, 7);
        let qs: Vec<&Vec<f32>> = data.iter().step_by(200).collect();
        let mut p_full = 0u64;
        let mut t_full = 0u64;
        for q in &qs {
            let (_, sp) = plain.search_with_stats(q, 10);
            let (_, st) = tri.search_with_stats(q, 10);
            p_full += sp.full_dist;
            t_full += st.full_dist;
        }
        // Tribase should evaluate strictly fewer full distances on
        // clustered data with self-queries (heap tightens to ~0
        // immediately).
        assert!(t_full < p_full, "plain={p_full} tri={t_full}");
    }

    #[test]
    fn tribase_top1_matches_flat_on_random_queries() {
        let data = make_clustered(2_000, 16, 10, 0.06, 17);
        let flat = FlatIndex::new(data.clone());
        // Probe enough clusters for high recall.
        let tri = TribaseIndex::build(data.clone(), 10, 10, 25, 19);
        let mut hits = 0;
        let mut total = 0;
        for (i, q) in data.iter().enumerate().step_by(53) {
            let a = flat.search(q, 1);
            let b = tri.search(q, 1);
            if a[0].id == b[0].id {
                hits += 1;
            }
            total += 1;
            // Sanity: don't infinite-loop on tiny test.
            if i > 1500 {
                break;
            }
        }
        // Probing all clusters → recall must be perfect.
        assert_eq!(hits, total);
    }
}
