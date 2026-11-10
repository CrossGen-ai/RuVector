//! Permutation-Prefix Reranker.
//!
//! For each vector we store the *ordered* ids of its `prefix_m` nearest
//! pivots, out of `n_pivots` total. At query time we compute the query's
//! own prefix once, then for each candidate reduce the full-footrule
//! distance to a simple prefix overlap: pivots appearing in both prefixes
//! contribute `|rank_q - rank_c|`, pivots in only one contribute a
//! fixed penalty `prefix_m` (approx. "beyond prefix").
//!
//! This is the *Spearman-footrule-at-top-m* distance of Chavez et al.
//! (2008), rediscovered as a cheap sidecar reranker.

use crate::{pivot, sq_l2, Reranker};
use std::time::Instant;

pub struct PermutationPrefixReranker {
    pub pivots: Vec<Vec<f32>>,
    pub n_pivots: usize,
    pub prefix_m: usize,
    pub codes: Vec<Vec<u16>>, // per-vector: top prefix_m pivot ids in rank order
    pub dim: usize,
}

impl PermutationPrefixReranker {
    pub fn build(
        vectors: &[Vec<f32>],
        n_pivots: usize,
        prefix_m: usize,
        seed: u64,
    ) -> (Self, std::time::Duration) {
        assert!(n_pivots <= u16::MAX as usize, "n_pivots must fit in u16");
        assert!(prefix_m <= n_pivots);
        let start = Instant::now();
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let pivots = pivot::farthest_point_sample(vectors, n_pivots, seed);
        let codes: Vec<Vec<u16>> = vectors
            .iter()
            .map(|v| prefix_for(v, &pivots, prefix_m))
            .collect();
        let elapsed = start.elapsed();
        (
            Self {
                pivots,
                n_pivots,
                prefix_m,
                codes,
                dim,
            },
            elapsed,
        )
    }

    /// Spearman-footrule-at-top-m distance using a precomputed query rank map.
    /// `q_rank[pid]` is the query's rank of pivot pid, or i32::MAX if pid
    /// is not in the query prefix.
    #[inline]
    pub fn footrule_with_map(&self, q_rank: &[i32], cand_prefix: &[u16]) -> u32 {
        let m = self.prefix_m as i32;
        let mut dist: i32 = 0;
        let mut shared: i32 = 0;
        for (rc, &pid) in cand_prefix.iter().enumerate() {
            let rq = q_rank[pid as usize];
            if rq != i32::MAX {
                dist += (rc as i32 - rq).abs();
                shared += 1;
            } else {
                dist += m;
            }
        }
        dist += (m - shared) * m;
        dist as u32
    }

    /// Spearman-footrule-at-top-m distance between two prefixes (convenience
    /// wrapper that builds the rank map internally; allocates — use
    /// `footrule_with_map` for hot loops).
    pub fn footrule(&self, query_prefix: &[u16], cand_prefix: &[u16]) -> u32 {
        let mut q_rank: Vec<i32> = vec![i32::MAX; self.n_pivots];
        for (r, &pid) in query_prefix.iter().enumerate() {
            q_rank[pid as usize] = r as i32;
        }
        self.footrule_with_map(&q_rank, cand_prefix)
    }
}

impl Reranker for PermutationPrefixReranker {
    fn name(&self) -> &'static str {
        "PermutationPrefix"
    }
    fn bytes_per_vector(&self) -> usize {
        self.prefix_m * std::mem::size_of::<u16>()
    }
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32> {
        let qp = prefix_for(query, &self.pivots, self.prefix_m);
        // Build the rank map once per query.
        let mut q_rank: Vec<i32> = vec![i32::MAX; self.n_pivots];
        for (r, &pid) in qp.iter().enumerate() {
            q_rank[pid as usize] = r as i32;
        }
        let mut scored: Vec<(u32, u32)> = candidates
            .iter()
            .map(|&id| {
                let d = self.footrule_with_map(&q_rank, &self.codes[id as usize]);
                (d, id)
            })
            .collect();
        scored.sort_by_key(|x| x.0);
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

/// Compute the top-`m` pivot ids for a vector, in ascending distance order.
pub fn prefix_for(v: &[f32], pivots: &[Vec<f32>], m: usize) -> Vec<u16> {
    let mut d: Vec<(f32, u16)> = pivots
        .iter()
        .enumerate()
        .map(|(i, p)| (sq_l2(v, p), i as u16))
        .collect();
    // Partial sort: we only need the top m.
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(m).map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sorts_pivots_by_distance() {
        let pivots = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];
        let v = vec![1.0, 0.0];
        let p = prefix_for(&v, &pivots, 3);
        assert_eq!(p[0], 0, "nearest pivot is origin");
        // Second nearest is (10,0): sq_l2 = 81 vs (0,10): sq_l2 = 101.
        assert_eq!(p[1], 1);
        assert_eq!(p[2], 2);
    }

    #[test]
    fn footrule_identical_prefix_is_zero() {
        let vs: Vec<Vec<f32>> = (0..64)
            .map(|i| vec![(i as f32).sin(), (i as f32).cos(), (i as f32) * 0.01])
            .collect();
        let (ppr, _) = PermutationPrefixReranker::build(&vs, 8, 4, 1);
        let code = ppr.codes[0].clone();
        assert_eq!(ppr.footrule(&code, &code), 0);
    }

    #[test]
    fn footrule_grows_with_mismatch() {
        let vs: Vec<Vec<f32>> = (0..64)
            .map(|i| vec![(i as f32).sin(), (i as f32).cos(), (i as f32) * 0.01])
            .collect();
        let (ppr, _) = PermutationPrefixReranker::build(&vs, 8, 4, 1);
        let a = vec![0u16, 1, 2, 3];
        let b = vec![0u16, 1, 2, 3];
        let c = vec![3u16, 2, 1, 0]; // reversed — maximum in-prefix footrule
        assert!(ppr.footrule(&a, &c) > ppr.footrule(&a, &b));
    }

    #[test]
    fn rerank_recovers_nearest_on_toy_clusters() {
        // Two tight clusters; query near cluster A should rerank cluster A ids first.
        let mut vs: Vec<Vec<f32>> = (0..32)
            .map(|i| vec![0.0 + (i as f32) * 0.001, 0.0])
            .collect();
        vs.extend((0..32).map(|i| vec![5.0 + (i as f32) * 0.001, 5.0]));
        let (ppr, _) = PermutationPrefixReranker::build(&vs, 6, 4, 2);
        let query = vec![0.0, 0.0];
        let cands: Vec<u32> = (0..64u32).collect();
        let top = ppr.rerank(&query, &cands, 8);
        // All top-8 should be from the first (cluster-A) 32 ids.
        assert!(top.iter().all(|&id| id < 32), "top = {top:?}");
    }
}
