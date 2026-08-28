//! Flat pivot index used to feed the FINGER estimators.
//!
//! For a pure FINGER benchmark we do not need a full HNSW graph — what matters
//! is that (a) each query lands on a pivot with a known dot-product, and
//! (b) each pivot has a stable list of "neighbours" we must score. A random
//! √n-sized pivot set with nearest-anchor assignment satisfies both while
//! remaining trivially reproducible.

use crate::{dot, Dataset};
use rand::prelude::*;
use rand::rngs::StdRng;

pub struct PivotIndex {
    pub dim: usize,
    /// Pivot vectors (unit-norm).
    pub pivots: Vec<Vec<f32>>,
    /// For each pivot: the ids of dataset vectors assigned to it.
    pub members: Vec<Vec<u32>>,
    /// For each vector id → assigned pivot id.
    pub assignment: Vec<u32>,
    /// For each vector id → `v · pivot_of(v)` (precomputed).
    pub anchor_dot: Vec<f32>,
    /// Owned copy of the dataset vectors (for exact fallback + residual builds).
    pub vectors: Vec<Vec<f32>>,
}

impl PivotIndex {
    pub fn build(ds: &Dataset, n_pivots: usize, seed: u64) -> Self {
        assert!(n_pivots > 0, "need at least one pivot");
        assert!(n_pivots <= ds.len(), "more pivots than vectors");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut idx: Vec<usize> = (0..ds.len()).collect();
        idx.shuffle(&mut rng);
        let pivots: Vec<Vec<f32>> = idx.iter().take(n_pivots).map(|&i| ds.vectors[i].clone()).collect();

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_pivots];
        let mut assignment = vec![0u32; ds.len()];
        let mut anchor_dot = vec![0f32; ds.len()];
        for (vid, v) in ds.vectors.iter().enumerate() {
            let mut best = (f32::NEG_INFINITY, 0u32);
            for (pid, p) in pivots.iter().enumerate() {
                let s = dot(v, p);
                if s > best.0 {
                    best = (s, pid as u32);
                }
            }
            assignment[vid] = best.1;
            anchor_dot[vid] = best.0;
            members[best.1 as usize].push(vid as u32);
        }
        Self {
            dim: ds.dim,
            pivots,
            members,
            assignment,
            anchor_dot,
            vectors: ds.vectors.clone(),
        }
    }

    /// Return the residual `v - (v·p) p` (unit-normalised anchor `p` assumed).
    pub fn residual(&self, vid: u32) -> Vec<f32> {
        let v = &self.vectors[vid as usize];
        let p = &self.pivots[self.assignment[vid as usize] as usize];
        let a = self.anchor_dot[vid as usize];
        let mut r = vec![0f32; self.dim];
        for i in 0..self.dim {
            r[i] = v[i] - a * p[i];
        }
        r
    }

    /// Query entry: pick the top-`beam` pivots by |q · pivot|. Returns
    /// `(pivot_id, q·pivot)` pairs sorted by descending inner product.
    pub fn entry(&self, query: &[f32], beam: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = self
            .pivots
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u32, dot(query, p)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(beam.max(1));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_vector_has_a_pivot() {
        let ds = Dataset::synthetic(300, 4, 32, 11);
        let idx = PivotIndex::build(&ds, 16, 3);
        let total: usize = idx.members.iter().map(|m| m.len()).sum();
        assert_eq!(total, ds.len());
        for a in &idx.assignment {
            assert!((*a as usize) < idx.pivots.len());
        }
    }

    #[test]
    fn residual_is_orthogonal_to_anchor() {
        let ds = Dataset::synthetic(64, 1, 24, 5);
        let idx = PivotIndex::build(&ds, 8, 1);
        for vid in 0..ds.len() as u32 {
            let r = idx.residual(vid);
            let p = &idx.pivots[idx.assignment[vid as usize] as usize];
            let s: f32 = r.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
            assert!(s.abs() < 1e-3, "residual not orthogonal: {s}");
        }
    }
}
