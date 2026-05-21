//! Optimized Product Quantization (Ge et al., CVPR 2013 / TPAMI 2014).
//!
//! Alternates:
//!   * Fix R, train per-subspace codebooks on rotated training data (plain PQ).
//!   * Fix codebooks, compute residuals Y = decode(encode(X R)); update R via
//!     orthogonal Procrustes on (X, Y).
//!
//! Iterating 3–5 sweeps typically converges. The rotation moves
//! energy between subspaces so the PQ assumption (subspaces are independent)
//! becomes a better fit.

use crate::error::Error;
use crate::metrics::sq_l2;
use crate::pq::Pq;
use crate::rotation::{matmul, procrustes, random_orthogonal, rotate, transpose, Matrix};
use crate::Quantizer;

#[derive(Clone)]
pub struct Opq {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub r: Matrix,        // d x d rotation
    pub r_t: Matrix,      // cached transpose
    pub inner: Pq,        // codebooks live in rotated space
}

impl Opq {
    pub fn train(
        train: &[Vec<f32>],
        m: usize,
        k: usize,
        kmeans_iters: usize,
        opq_sweeps: usize,
        seed: u64,
    ) -> Result<Self, Error> {
        let d = train.first().map(|v| v.len()).unwrap_or(0);
        let mut r = random_orthogonal(d, seed);

        // initial: train Pq on rotated data
        let rotated: Vec<Vec<f32>> = train
            .iter()
            .map(|x| {
                let mut o = vec![0f32; d];
                rotate(&r, x, &mut o);
                o
            })
            .collect();
        let mut pq = Pq::train(&rotated, m, k, kmeans_iters, seed)?;

        for sweep in 0..opq_sweeps {
            // build Y = decode(encode(X R)) in rotated space
            let mut y = vec![vec![0f32; d]; train.len()];
            let mut x_rot = vec![vec![0f32; d]; train.len()];
            for (i, v) in train.iter().enumerate() {
                rotate(&r, v, &mut x_rot[i]);
                let code = pq.encode(&x_rot[i]);
                for s in 0..m {
                    let off = s * pq.sub_dim;
                    let c = code[s] as usize;
                    y[i][off..off + pq.sub_dim].copy_from_slice(&pq.codebooks[s][c]);
                }
            }
            // build X^T Y (d x d): X is train (n x d), Y is in rotated space (n x d).
            // Procrustes minimizes ||X R - Y||_F, so we want X^T Y.
            let mut xty = vec![vec![0f32; d]; d];
            for i in 0..train.len() {
                let xi = &train[i];
                let yi = &y[i];
                for a in 0..d {
                    let xa = xi[a];
                    if xa == 0.0 {
                        continue;
                    }
                    for b in 0..d {
                        xty[a][b] += xa * yi[b];
                    }
                }
            }
            r = procrustes(&xty);

            // retrain codebooks under new R
            let rotated: Vec<Vec<f32>> = train
                .iter()
                .map(|x| {
                    let mut o = vec![0f32; d];
                    rotate(&r, x, &mut o);
                    o
                })
                .collect();
            pq = Pq::train(&rotated, m, k, kmeans_iters, seed.wrapping_add(31 * (sweep as u64 + 1)))?;
        }

        let r_t = transpose(&r);
        // sanity: R R^T ≈ I (we don't error if jacobi drifted slightly)
        let _ = matmul(&r, &r_t);

        Ok(Self { d, m, k, r, r_t, inner: pq })
    }
}

impl Quantizer for Opq {
    fn encode(&self, x: &[f32]) -> Vec<u8> {
        let mut rx = vec![0f32; self.d];
        rotate(&self.r, x, &mut rx);
        self.inner.encode(&rx)
    }

    fn asymmetric_score(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut rq = vec![0f32; self.d];
        rotate(&self.r, query, &mut rq);
        self.inner.asymmetric_score(&rq, code)
    }

    fn code_bytes(&self) -> usize {
        self.inner.code_bytes()
    }

    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }
}

#[allow(dead_code)]
fn check_orthonormal(r: &Matrix) -> f32 {
    let d = r.len();
    let rt = transpose(r);
    let rrt = matmul(r, &rt);
    let mut err = 0f32;
    for i in 0..d {
        for j in 0..d {
            let target = if i == j { 1.0 } else { 0.0 };
            let diff = rrt[i][j] - target;
            err += diff * diff;
        }
    }
    err.sqrt()
}

#[allow(dead_code)]
fn approx_l2_under_decode(x: &[f32], code: &[u8], pq: &Pq) -> f32 {
    let mut decoded = vec![0f32; pq.d];
    for s in 0..pq.m {
        let off = s * pq.sub_dim;
        let c = code[s] as usize;
        decoded[off..off + pq.sub_dim].copy_from_slice(&pq.codebooks[s][c]);
    }
    sq_l2(x, &decoded)
}
