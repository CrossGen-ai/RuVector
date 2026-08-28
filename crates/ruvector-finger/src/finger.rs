//! FINGER estimator with per-pivot PCA basis.
//!
//! For each pivot `p` we collect the residuals of its assigned members, run
//! deflated power iteration to extract the top-`rank` eigenvectors of the
//! residual covariance, and store each residual as a small code in that basis.
//!
//! Scoring for a candidate neighbour `v` assigned to pivot `p`:
//!
//!   score(q, v) ≈ (q · p) · (v · p) + <B_p^T q_res, code_v>
//!
//! `B_p` is stored as `rank` unit vectors of length `d` per pivot. Query prep
//! projects `q_res = q − (q · p) p` onto `B_p` once, then each neighbour is a
//! rank-length dot product — the FINGER speed win.

use crate::estimator::{DistanceEstimator, PivotHandle};
use crate::{normalise, PivotIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

pub struct FingerEstimator<'a> {
    idx: &'a PivotIndex,
    pub rank: usize,
    /// Per-pivot basis: `basis[pid][k]` is a length-`d` unit vector.
    basis: Vec<Vec<Vec<f32>>>,
    /// Per-vector code (in the basis of its assigned pivot).
    codes: Vec<Vec<f32>>,
    name: &'static str,
}

impl<'a> FingerEstimator<'a> {
    pub fn build(idx: &'a PivotIndex, rank: usize, seed: u64, name: &'static str) -> Self {
        let n_pivots = idx.pivots.len();
        let mut basis: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_pivots];
        let mut codes = vec![Vec::new(); idx.vectors.len()];
        let mut rng = StdRng::seed_from_u64(seed);

        for pid in 0..n_pivots {
            let members = &idx.members[pid];
            if members.is_empty() {
                basis[pid] = (0..rank).map(|_| vec![0f32; idx.dim]).collect();
                continue;
            }
            // Gather residuals for this pivot.
            let residuals: Vec<Vec<f32>> = members.iter().map(|&vid| idx.residual(vid)).collect();
            let b = pca_top_r(&residuals, idx.dim, rank, &mut rng);
            for (i, vid) in members.iter().enumerate() {
                let mut c = Vec::with_capacity(rank);
                for k in 0..rank {
                    let mut s = 0f32;
                    for j in 0..idx.dim { s += b[k][j] * residuals[i][j]; }
                    c.push(s);
                }
                codes[*vid as usize] = c;
            }
            basis[pid] = b;
        }
        Self { idx, rank, basis, codes, name }
    }
}

struct FingerHandle<'a> {
    est: &'a FingerEstimator<'a>,
    q_anchor: f32,
    q_proj: Vec<f32>,
}

impl<'a> PivotHandle for FingerHandle<'a> {
    fn score(&self, neighbour: u32) -> f32 {
        let a_v = self.est.idx.anchor_dot[neighbour as usize];
        let mut s = self.q_anchor * a_v;
        let code = &self.est.codes[neighbour as usize];
        // `code` may live in a *different* pivot's basis than the query's
        // pivot handle — that is FINGER's original approximation. When the
        // query pivot equals the neighbour's home pivot the approximation is
        // consistent; otherwise we still gain an anchor-based lower bound.
        for k in 0..self.est.rank.min(code.len()) {
            s += self.q_proj[k] * code[k];
        }
        s
    }
}

impl<'a> DistanceEstimator for FingerEstimator<'a> {
    fn name(&self) -> &'static str { self.name }
    fn bytes_per_vector(&self) -> usize { self.rank * std::mem::size_of::<f32>() }

    fn prepare_query<'b>(&'b self, query: &[f32], pivot_id: u32) -> Box<dyn PivotHandle + 'b> {
        let p = &self.idx.pivots[pivot_id as usize];
        let mut q_anchor = 0f32;
        for i in 0..self.idx.dim { q_anchor += query[i] * p[i]; }
        let mut q_res = vec![0f32; self.idx.dim];
        for i in 0..self.idx.dim { q_res[i] = query[i] - q_anchor * p[i]; }
        let b = &self.basis[pivot_id as usize];
        let mut q_proj = Vec::with_capacity(self.rank);
        for bk in b {
            let mut s = 0f32;
            for i in 0..self.idx.dim { s += bk[i] * q_res[i]; }
            q_proj.push(s);
        }
        Box::new(FingerHandle { est: self, q_anchor, q_proj })
    }
}

/// Deflated power iteration for the top-`rank` eigenvectors of the empirical
/// residual covariance. Implicitly computed via samples — never materialise
/// the d×d matrix (d can be ≥ 128).
fn pca_top_r(samples: &[Vec<f32>], dim: usize, rank: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let iters = 20;
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(rank);
    let mut work: Vec<Vec<f32>> = samples.to_vec();

    for _ in 0..rank {
        // Random unit init.
        let mut b: Vec<f32> = (0..dim).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        normalise(&mut b);

        for _ in 0..iters {
            // y = Σ_i (x_i · b) x_i
            let mut y = vec![0f32; dim];
            for x in &work {
                let mut s = 0f32;
                for j in 0..dim { s += x[j] * b[j]; }
                for j in 0..dim { y[j] += s * x[j]; }
            }
            normalise(&mut y);
            b = y;
        }

        // Deflate: remove component along b from every sample.
        for x in work.iter_mut() {
            let mut s = 0f32;
            for j in 0..dim { s += x[j] * b[j]; }
            for j in 0..dim { x[j] -= s * b[j]; }
        }
        basis.push(b);
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn pca_basis_is_unit_norm() {
        let ds = Dataset::synthetic(400, 4, 32, 9);
        let idx = PivotIndex::build(&ds, 16, 2);
        let est = FingerEstimator::build(&idx, 8, 3, "finger-r8");
        for basis in &est.basis {
            for b in basis {
                let n: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((n - 1.0).abs() < 1e-3 || n < 1e-6);
            }
        }
    }
}
