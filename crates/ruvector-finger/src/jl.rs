//! JL (Johnson–Lindenstrauss) baseline: a single global Gaussian projection
//! `B ∈ R^{d×r}` shared across all pivots. Codes are `B^T v_res`. Scoring:
//!
//!   q·v ≈ (q·pivot)(v·pivot) + <B^T q_res, code_v>
//!
//! No per-pivot adaptation, so it acts as a fair "dumb" low-rank comparison to
//! FINGER's per-pivot PCA basis.

use crate::estimator::{DistanceEstimator, PivotHandle};
use crate::{PivotIndex, gaussian_unit};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct JlEstimator<'a> {
    idx: &'a PivotIndex,
    rank: usize,
    /// `basis[k]` is the k-th projection direction, length = d.
    basis: Vec<Vec<f32>>,
    /// Per-vector code of length `rank` (in the *global* basis).
    codes: Vec<Vec<f32>>,
    name: &'static str,
}

impl<'a> JlEstimator<'a> {
    pub fn build(idx: &'a PivotIndex, rank: usize, seed: u64, name: &'static str) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Sample rank orthonormal-ish directions. For the sizes we care about
        // (r ≤ 32) simple Gaussian directions are near-orthogonal by JL theory
        // — sufficient for a baseline. We normalise each row to unit length.
        let basis: Vec<Vec<f32>> = (0..rank).map(|_| gaussian_unit(&mut rng, idx.dim)).collect();

        let n = idx.vectors.len();
        let mut codes = vec![Vec::with_capacity(rank); n];
        for vid in 0..n {
            let r = idx.residual(vid as u32);
            let mut c = Vec::with_capacity(rank);
            for b in &basis {
                let mut s = 0f32;
                for i in 0..idx.dim { s += b[i] * r[i]; }
                c.push(s);
            }
            codes[vid] = c;
        }
        Self { idx, rank, basis, codes, name }
    }
}

struct JlHandle<'a> {
    est: &'a JlEstimator<'a>,
    q_anchor: f32,
    q_proj: Vec<f32>,
}

impl<'a> PivotHandle for JlHandle<'a> {
    fn score(&self, neighbour: u32) -> f32 {
        let a_v = self.est.idx.anchor_dot[neighbour as usize];
        let mut s = self.q_anchor * a_v;
        let code = &self.est.codes[neighbour as usize];
        for k in 0..self.est.rank {
            s += self.q_proj[k] * code[k];
        }
        s
    }
}

impl<'a> DistanceEstimator for JlEstimator<'a> {
    fn name(&self) -> &'static str { self.name }
    fn bytes_per_vector(&self) -> usize { self.rank * std::mem::size_of::<f32>() }
    fn prepare_query<'b>(&'b self, query: &[f32], pivot_id: u32) -> Box<dyn PivotHandle + 'b> {
        let p = &self.idx.pivots[pivot_id as usize];
        // q · pivot and residual q_res = q - (q·p) p
        let mut q_anchor = 0f32;
        for i in 0..self.idx.dim { q_anchor += query[i] * p[i]; }
        let mut q_res = vec![0f32; self.idx.dim];
        for i in 0..self.idx.dim { q_res[i] = query[i] - q_anchor * p[i]; }
        // Project q_res into the global basis.
        let mut q_proj = Vec::with_capacity(self.rank);
        for b in &self.basis {
            let mut s = 0f32;
            for i in 0..self.idx.dim { s += b[i] * q_res[i]; }
            q_proj.push(s);
        }
        Box::new(JlHandle { est: self, q_anchor, q_proj })
    }
}
