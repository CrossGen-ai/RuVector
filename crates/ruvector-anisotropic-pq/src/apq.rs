//! Anisotropic Product Quantization (ScaNN, Guo et al. ICML 2020).
//!
//! Plain PQ minimizes the spherical L2 error. For maximum inner product
//! search the quantity that matters is the *parallel* component of the
//! residual `r = x − c` (the part along x), because that's what biases
//! inner-product estimates. The anisotropic loss replaces spherical L2 with
//! a weighted form: parallel-residual error is multiplied by `h_parallel`
//! and orthogonal-residual error by `h_orth`, with `h_parallel >> h_orth`.
//! The ratio `eta = h_parallel / h_orth` is the anisotropy factor
//! (`eta = 1` collapses to plain PQ; `eta ≈ d - 1` is the theoretical
//! optimum cited in the paper for top-k MIPS).
//!
//! Implementation: iterative weighted-k-means. After plain-PQ init, we do a
//! few refinement sweeps; on each sweep, for each training point i we
//! compute a per-point weight from the alignment of its current residual
//! with `x_i` (high cos² → emphasize parallel error → larger weight), then
//! rerun weighted Lloyd in every subspace. This is an approximation of the
//! per-subspace anisotropic objective; it is NOT a line-for-line port of
//! ScaNN's exact closed-form (which uses subspace-conditional norms).
//! Treat it as anisotropic-style weighted PQ — judge it by the reproducible
//! recall numbers in the demo binary, not by the citation alone.

use crate::error::Error;
use crate::kmeans::lloyd;
use crate::metrics::{dot, norm2, sq_l2};
use crate::pq::Pq;
use crate::Quantizer;

#[derive(Clone)]
pub struct AnisotropicPq {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
    pub eta: f32,
    /// codebooks[s][c] — same shape as plain Pq.
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl AnisotropicPq {
    pub fn train(
        train: &[Vec<f32>],
        m: usize,
        k: usize,
        eta: f32,
        kmeans_iters: usize,
        refinement_sweeps: usize,
        seed: u64,
    ) -> Result<Self, Error> {
        // 1. plain PQ as initialization (gives us an honest fallback if
        //    sweeps don't help)
        let pq0 = Pq::train(train, m, k, kmeans_iters, seed)?;
        let d = pq0.d;
        let sub_dim = pq0.sub_dim;
        let mut codebooks = pq0.codebooks.clone();
        let h_orth = 1.0f32;
        let h_par = h_orth * eta.max(1.0);

        // current PQ to use for residual computation
        let mut current = Pq { d, m, k, sub_dim, codebooks: codebooks.clone() };

        for _sweep in 0..refinement_sweeps {
            // decode current codes and compute residuals + per-point weights
            let mut weights = Vec::with_capacity(train.len());
            for x in train {
                let code = current.encode(x);
                let mut decoded = vec![0f32; d];
                for s in 0..m {
                    let off = s * sub_dim;
                    let c = code[s] as usize;
                    decoded[off..off + sub_dim].copy_from_slice(&current.codebooks[s][c]);
                }
                let mut r = vec![0f32; d];
                for i in 0..d {
                    r[i] = x[i] - decoded[i];
                }
                let nx = norm2(x);
                let nr = norm2(&r);
                if nx <= 1e-12 || nr <= 1e-12 {
                    weights.push(1.0);
                    continue;
                }
                let xr = dot(x, &r);
                // cos² of angle between r and x
                let cos2 = (xr * xr) / (nx * nr);
                let w = h_orth + (h_par - h_orth) * cos2;
                weights.push(w);
            }

            // retrain each subspace with weighted Lloyd
            for s in 0..m {
                let sub: Vec<Vec<f32>> = train
                    .iter()
                    .map(|v| v[s * sub_dim..(s + 1) * sub_dim].to_vec())
                    .collect();
                let r = lloyd(
                    &sub,
                    Some(&weights),
                    k,
                    kmeans_iters,
                    seed.wrapping_add(1009 * (s as u64 + 1)),
                );
                codebooks[s] = r.centroids;
            }
            current = Pq { d, m, k, sub_dim, codebooks: codebooks.clone() };
        }

        Ok(Self { d, m, k, sub_dim, eta, codebooks })
    }

    /// Asymmetric score for MIPS: returns NEGATIVE inner product so that
    /// "smaller is better" matches the L2 backends. Caller can negate at the
    /// top of the ranking loop if they want a max-IP score.
    pub fn neg_inner_product(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut s = 0f32;
        for sub in 0..self.m {
            let q = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let c = code[sub] as usize;
            s += dot(q, &self.codebooks[sub][c]);
        }
        -s
    }

    /// Build a (m * k) lookup table of inner products so per-database-vector
    /// scoring is m table-lookups (the standard PQ scan kernel).
    pub fn ip_lookup_table(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let q = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            for c in 0..self.k {
                lut[s * self.k + c] = dot(q, &self.codebooks[s][c]);
            }
        }
        lut
    }
}

impl Quantizer for AnisotropicPq {
    fn encode(&self, x: &[f32]) -> Vec<u8> {
        // L2-nearest in each subspace, same as plain PQ; the weighting changed
        // *training*, not encoding.
        let mut code = vec![0u8; self.m];
        for s in 0..self.m {
            let sub = &x[s * self.sub_dim..(s + 1) * self.sub_dim];
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let d = sq_l2(sub, &self.codebooks[s][c]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            code[s] = best as u8;
        }
        code
    }

    fn asymmetric_score(&self, query: &[f32], code: &[u8]) -> f32 {
        self.neg_inner_product(query, code)
    }

    fn code_bytes(&self) -> usize {
        self.m
    }

    fn shape(&self) -> (usize, usize, usize) {
        (self.m, self.d, self.k)
    }
}
