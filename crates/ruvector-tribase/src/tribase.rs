//! Tribase: triangle-inequality pivot pruning for IVF cluster search.
//!
//! Reference: Liu et al., "Triangle-Inequality-Based Pruning for IVF Search",
//! SIGMOD 2024 (the core idea; this is an independent CPU Rust implementation).
//!
//! ## Idea
//!
//! Let `c` be the centroid of an IVF list, and let `r(x)` be the L2 distance
//! from a vector `x` in that list to `c` (precomputed at build time). For any
//! query `q`, by the triangle inequality:
//!
//! ```text
//! | d(q,c) - r(x) | <= d(q,x) <= d(q,c) + r(x)
//! ```
//!
//! Let `tau` be the current k-th-best L2 distance during search. Then:
//!
//!   * (LB pruning) If `d(q,c) - r(x) > tau`, x cannot improve top-k. Skip.
//!   * (UB acceptance) If `d(q,c) + r(x) < tau`, x is guaranteed top-k --
//!     we still must compute exact distance to rank it correctly, but the
//!     visit is rare and dominated by the LB path.
//!
//! ## Implementation
//!
//! For each cluster we sort `(r(x), x_id)` ascending. Tribase scans the list
//! and maintains running `d(q,c)`. The LB increases with `|r(x) - d(q,c)|`,
//! so we can stop early once even the closest unseen r(x) cannot beat tau.
//! Concretely, we split each list into two halves at the index where
//! `r(x) >= d(q,c)`:
//!
//! - Lower half (r(x) &lt; d(q,c)): LB = d(q,c) - r(x), decreases as r(x)
//!   grows; scan from the split downward (largest r first), break when
//!   LB &gt; tau.
//! - Upper half (r(x) &gt;= d(q,c)): LB = r(x) - d(q,c), increases as r(x)
//!   grows; scan from the split upward, break when LB &gt; tau.
//!
//! Vectors that pass the LB test still get one exact distance computation;
//! everything else is skipped. We track stats for the benchmark harness.

use crate::dist::{l2, sq_l2, TopK};
use crate::flat::SearchStats;
use crate::kmeans;

pub struct TribaseIndex {
    pub centroids: Vec<Vec<f32>>,
    /// Per-list: ids sorted ascending by residual r(x)
    pub list_ids: Vec<Vec<u32>>,
    /// Per-list: residuals r(x) (L2, not squared) parallel to list_ids
    pub list_residuals: Vec<Vec<f32>>,
    pub data: Vec<Vec<f32>>,
    pub dim: usize,
}

impl TribaseIndex {
    pub fn build(
        data: Vec<Vec<f32>>,
        n_lists: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        let dim = data[0].len();
        let km = kmeans::fit(&data, n_lists, kmeans_iters, seed);

        // Bucket vectors into lists with their residuals
        let mut buckets: Vec<Vec<(f32, u32)>> = vec![Vec::new(); n_lists];
        for (j, &c) in km.assignments.iter().enumerate() {
            let r = l2(&data[j], &km.centroids[c as usize]);
            buckets[c as usize].push((r, j as u32));
        }
        // Sort each list ascending by residual
        let mut list_ids: Vec<Vec<u32>> = Vec::with_capacity(n_lists);
        let mut list_residuals: Vec<Vec<f32>> = Vec::with_capacity(n_lists);
        for mut b in buckets {
            b.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (rs, ids): (Vec<_>, Vec<_>) = b.into_iter().unzip();
            list_residuals.push(rs);
            list_ids.push(ids);
        }

        Self {
            centroids: km.centroids,
            list_ids,
            list_residuals,
            data,
            dim,
        }
    }

    /// Search with triangle-inequality pruning.
    /// `nprobe` = number of nearest centroid lists to scan.
    pub fn search(&self, q: &[f32], k: usize, nprobe: usize) -> (Vec<(f32, u32)>, SearchStats) {
        let mut stats = SearchStats::default();

        // Squared and exact distance to each centroid
        let mut cd: Vec<(f32, f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let sq = sq_l2(q, c);
                (sq, sq.sqrt(), i)
            })
            .collect();
        stats.dist_computations += self.centroids.len() as u64;
        cd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let probe = nprobe.min(cd.len());

        let mut heap = TopK::new(k);

        for &(_sq, dqc, ci) in cd[..probe].iter() {
            stats.lists_scanned += 1;
            let ids = &self.list_ids[ci];
            let rs = &self.list_residuals[ci];
            if ids.is_empty() {
                continue;
            }
            // Bisect: smallest idx where rs[idx] >= dqc
            let split = rs.partition_point(|&r| r < dqc);

            // Lower half: indices [0..split), scan from split-1 downward.
            // For these r(x) < dqc, LB = dqc - r(x). As we walk left, r(x)
            // decreases, so LB grows. Once LB > tau_l2_sqrt we can stop.
            // tau is squared distance kept by heap; compare via sqrt once.
            // We carry tau_sqrt cached and refresh only on improvement.
            let mut tau_sq = heap.worst();
            let mut tau = tau_sq.sqrt();
            if split > 0 {
                let mut i = split;
                while i > 0 {
                    i -= 1;
                    let lb = dqc - rs[i];
                    if lb > tau {
                        break;
                    }
                    let id = ids[i];
                    let d = sq_l2(q, &self.data[id as usize]);
                    stats.dist_computations += 1;
                    stats.vectors_scanned += 1;
                    heap.push(d, id);
                    if d < tau_sq {
                        tau_sq = heap.worst();
                        tau = tau_sq.sqrt();
                    }
                }
            }

            // Upper half: indices [split..len), scan upward.
            // LB = r(x) - dqc, increases as r(x) grows; stop when > tau.
            let mut j = split;
            while j < rs.len() {
                let lb = rs[j] - dqc;
                if lb > tau {
                    break;
                }
                let id = ids[j];
                let d = sq_l2(q, &self.data[id as usize]);
                stats.dist_computations += 1;
                stats.vectors_scanned += 1;
                heap.push(d, id);
                if d < tau_sq {
                    tau_sq = heap.worst();
                    tau = tau_sq.sqrt();
                }
                j += 1;
            }
        }
        (heap.into_sorted(), stats)
    }
}
