//! Neighbor-selection strategies.
//!
//! Each pruner takes:
//!   - `pivot`: the vector whose neighbor list we are building
//!   - `candidates`: (id, distance-to-pivot) sorted ascending
//!   - `vectors`: full vector store (for pairwise checks)
//!   - `m`: max out-degree
//!
//! and returns up to `m` ids to keep as out-edges.

use crate::dist2;

pub trait Pruner: Send + Sync {
    fn name(&self) -> &'static str;
    fn select(
        &self,
        pivot: usize,
        candidates: &[(usize, f32)],
        vectors: &[Vec<f32>],
        m: usize,
    ) -> Vec<usize>;
}

/// Trivial baseline: keep the M nearest candidates.
pub struct Naive;

impl Pruner for Naive {
    fn name(&self) -> &'static str { "naive-topM" }
    fn select(&self, _pivot: usize, candidates: &[(usize, f32)], _vecs: &[Vec<f32>], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(i, _)| *i).collect()
    }
}

/// Relative Neighborhood Graph pruning (Malkov & Yashunin, HNSW). Given a
/// candidate `c` with distance `d(p, c)`, reject if there exists an already-
/// kept neighbor `n` with `d(n, c) < d(p, c)`. This enforces the RNG
/// property and produces diverse, long-range edges.
pub struct RngPrune;

impl Pruner for RngPrune {
    fn name(&self) -> &'static str { "rng" }
    fn select(&self, _pivot: usize, candidates: &[(usize, f32)], vectors: &[Vec<f32>], m: usize) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::with_capacity(m);
        for &(cid, dpc) in candidates {
            if kept.len() == m { break; }
            let mut dominated = false;
            for &nid in &kept {
                let dnc = dist2(&vectors[nid], &vectors[cid]);
                if dnc < dpc {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                kept.push(cid);
            }
        }
        kept
    }
}

/// Vamana / DiskANN α-pruning. Same shape as RNG but with a slack factor
/// `alpha ≥ 1`: accept `c` unless `alpha * d(n, c) < d(p, c)`. `alpha=1`
/// reduces to RNG. `alpha>1` keeps more edges → higher recall, more storage.
pub struct AlphaPrune {
    pub alpha: f32,
}

impl AlphaPrune {
    pub fn new(alpha: f32) -> Self { Self { alpha: alpha.max(1.0) } }
}

impl Pruner for AlphaPrune {
    fn name(&self) -> &'static str { "alpha-vamana" }
    fn select(&self, _pivot: usize, candidates: &[(usize, f32)], vectors: &[Vec<f32>], m: usize) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::with_capacity(m);
        for &(cid, dpc) in candidates {
            if kept.len() == m { break; }
            let mut dominated = false;
            for &nid in &kept {
                let dnc = dist2(&vectors[nid], &vectors[cid]);
                if self.alpha * dnc < dpc {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                kept.push(cid);
            }
        }
        kept
    }
}
