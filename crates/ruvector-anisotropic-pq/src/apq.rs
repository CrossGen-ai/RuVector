//! Anisotropic Product Quantization (score-aware loss).
//!
//! For each datapoint `x` with subvector `x_s` in subspace `s` and current
//! centroid `c`, the residual `r = x_s - c` is decomposed against the
//! sub-vector direction `u = x_s / ||x_s||`:
//!
//! ```text
//! r_parallel  = (r . u) * u
//! r_orthogonal = r - r_parallel
//! ```
//!
//! The anisotropic per-point loss is
//!
//! ```text
//! L = eta * ||r_parallel||^2 + ||r_orthogonal||^2
//!   = (eta - 1) * (r . u)^2 + ||r||^2
//! ```
//!
//! Setting `eta = 1` recovers standard k-means (MSE). For `eta > 1` the
//! gradient gives a closed-form update for the centroid `c`:
//!
//! ```text
//! d L / d c = -2 [(eta - 1) (r . u) u  + r]
//!           = -2 [ x - c + (eta - 1) u u^T (x - c) ]
//!           = -2 (I + (eta - 1) u u^T) (x - c)
//! ```
//!
//! Define `M_i = I + (eta - 1) u_i u_i^T` (rank-1 update; symmetric PSD).
//! Setting the sum to zero across the cluster gives the weighted least
//! squares centroid:
//!
//! ```text
//! c* = ( Sum_i M_i )^-1 ( Sum_i M_i x_i )
//! ```
//!
//! `Sum_i M_i = N I + (eta - 1) Sum_i u_i u_i^T`. We solve this small
//! `sub_dim x sub_dim` linear system per centroid per iteration. For the
//! typical PQ regime `sub_dim` is 4–16, which is trivial.

use crate::pq::{dot, sq_l2};
use crate::{QuantError, Quantizer};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct Apq {
    dim: usize,
    m: usize,
    k: usize,
    sub_dim: usize,
    pub eta: f32,
    pub(crate) codebooks: Vec<Vec<f32>>,
}

impl Apq {
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        eta: f32,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self, QuantError> {
        if data.is_empty() {
            return Err(QuantError::EmptyTraining);
        }
        if !(1..=256).contains(&k) {
            return Err(QuantError::InvalidK { k });
        }
        let dim = data[0].len();
        if dim % m != 0 {
            return Err(QuantError::SubspaceMismatch { dim, m });
        }
        let sub_dim = dim / m;

        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            let cb = train_anisotropic_subspace(
                data,
                s,
                sub_dim,
                k,
                eta,
                max_iter,
                seed.wrapping_add(s as u64),
            );
            codebooks.push(cb);
        }
        Ok(Self { dim, m, k, sub_dim, eta, codebooks })
    }
}

fn train_anisotropic_subspace(
    data: &[Vec<f32>],
    s: usize,
    sub_dim: usize,
    k: usize,
    eta: f32,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);

    // Initialise centroids from random points (same scheme as Pq).
    let mut centroids: Vec<f32> = Vec::with_capacity(k * sub_dim);
    for &i in idx.iter().take(k) {
        let v = &data[i][s * sub_dim..(s + 1) * sub_dim];
        centroids.extend_from_slice(v);
    }
    while centroids.len() < k * sub_dim {
        let last = centroids.len() - sub_dim;
        let copy: Vec<f32> = centroids[last..last + sub_dim].to_vec();
        centroids.extend_from_slice(&copy);
    }

    // Pre-compute unit sub-vectors (skip near-zero sub-vectors).
    let mut units: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);
    let mut subvecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for v in data {
        let sv = v[s * sub_dim..(s + 1) * sub_dim].to_vec();
        let nrm = sv.iter().map(|x| x * x).sum::<f32>().sqrt();
        let u = if nrm > 1e-12 {
            Some(sv.iter().map(|x| x / nrm).collect())
        } else {
            None
        };
        units.push(u);
        subvecs.push(sv);
    }

    let mut assign = vec![0u32; n];
    let lambda = eta - 1.0;

    for _iter in 0..max_iter {
        // ---- Assignment step: anisotropic distance ----
        let mut moved = 0usize;
        for i in 0..n {
            let sv = &subvecs[i];
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cv = &centroids[c * sub_dim..(c + 1) * sub_dim];
                // residual r = sv - cv
                let mut r = vec![0f32; sub_dim];
                for j in 0..sub_dim { r[j] = sv[j] - cv[j]; }
                let ru = match &units[i] {
                    Some(u) => dot(&r, u),
                    None => 0.0,
                };
                let d = sq_l2(sv, cv) + lambda * ru * ru;
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            if assign[i] != best {
                moved += 1;
                assign[i] = best;
            }
        }

        // ---- Update step: weighted least squares per centroid ----
        // For each centroid c we solve  A_c * c* = b_c  where
        //   A_c = N_c * I + lambda * Sum_{i in cluster c} u_i u_i^T
        //   b_c = Sum_{i in cluster c}  (I + lambda u_i u_i^T) x_i
        //       = Sum x_i + lambda * Sum (x_i . u_i) u_i
        let mut a_mats = vec![vec![0f32; sub_dim * sub_dim]; k];
        let mut b_vecs = vec![vec![0f32; sub_dim]; k];
        let mut counts = vec![0u32; k];

        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let sv = &subvecs[i];
            // b += sv
            for j in 0..sub_dim { b_vecs[c][j] += sv[j]; }
            if let Some(u) = &units[i] {
                let xu = dot(sv, u);
                for j in 0..sub_dim {
                    b_vecs[c][j] += lambda * xu * u[j];
                    for l in 0..sub_dim {
                        a_mats[c][j * sub_dim + l] += lambda * u[j] * u[l];
                    }
                }
            }
        }
        for c in 0..k {
            // Add N_c * I to A_c.
            if counts[c] == 0 { continue; }
            for j in 0..sub_dim {
                a_mats[c][j * sub_dim + j] += counts[c] as f32;
            }
            // Solve A_c x = b_c by Gauss-Jordan in place.
            let new_c = solve(&mut a_mats[c], &mut b_vecs[c], sub_dim);
            for j in 0..sub_dim {
                centroids[c * sub_dim + j] = new_c[j];
            }
        }

        if moved == 0 { break; }
    }
    centroids
}

/// In-place Gauss-Jordan solve of `A x = b`. A is `n x n` row-major.
/// Returns `x` (consumes b). Falls back to b on numerical failure.
fn solve(a: &mut [f32], b: &mut [f32], n: usize) -> Vec<f32> {
    // Augment [A | b] and reduce.
    let mut m = vec![0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n { m[i * (n + 1) + j] = a[i * n + j]; }
        m[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = m[r * (n + 1) + col].abs();
            if v > best { best = v; piv = r; }
        }
        if best < 1e-9 {
            // Singular -> return b as fallback.
            return b.to_vec();
        }
        if piv != col {
            for j in 0..=n {
                m.swap(col * (n + 1) + j, piv * (n + 1) + j);
            }
        }
        let div = m[col * (n + 1) + col];
        for j in 0..=n { m[col * (n + 1) + j] /= div; }
        for r in 0..n {
            if r == col { continue; }
            let factor = m[r * (n + 1) + col];
            if factor.abs() < 1e-30 { continue; }
            for j in 0..=n {
                m[r * (n + 1) + j] -= factor * m[col * (n + 1) + j];
            }
        }
    }
    (0..n).map(|i| m[i * (n + 1) + n]).collect()
}

impl Quantizer for Apq {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        // Anisotropic assignment, mirroring the training-time loss so the
        // codebook learned with score-aware weighting is also queried with
        // score-aware nearest-centroid lookup.
        let mut code = vec![0u8; self.m];
        let lambda = self.eta - 1.0;
        for s in 0..self.m {
            let sv = &v[s * self.sub_dim..(s + 1) * self.sub_dim];
            // Unit sub-vector direction (skip if near-zero).
            let nrm2: f32 = sv.iter().map(|x| x * x).sum();
            let unit: Option<Vec<f32>> = if nrm2 > 1e-24 {
                let n = nrm2.sqrt();
                Some(sv.iter().map(|x| x / n).collect())
            } else {
                None
            };
            let cb = &self.codebooks[s];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cv = &cb[c * self.sub_dim..(c + 1) * self.sub_dim];
                let mut d = sq_l2(sv, cv);
                if let Some(u) = &unit {
                    // r . u where r = sv - cv  =  (sv . u) - (cv . u)
                    let mut ru = 0.0f32;
                    for j in 0..self.sub_dim {
                        ru += (sv[j] - cv[j]) * u[j];
                    }
                    d += lambda * ru * ru;
                }
                if d < best_d {
                    best_d = d;
                    best = c as u8;
                }
            }
            code[s] = best;
        }
        code
    }

    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for s in 0..self.m {
            let c = code[s] as usize;
            let cv = &self.codebooks[s][c * self.sub_dim..(c + 1) * self.sub_dim];
            v[s * self.sub_dim..(s + 1) * self.sub_dim].copy_from_slice(cv);
        }
        v
    }

    fn m(&self) -> usize { self.m }
    fn k(&self) -> usize { self.k }

    fn dot_table(&self, q: &[f32]) -> Vec<f32> {
        let mut t = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let qs = &q[s * self.sub_dim..(s + 1) * self.sub_dim];
            let cb = &self.codebooks[s];
            for c in 0..self.k {
                let cv = &cb[c * self.sub_dim..(c + 1) * self.sub_dim];
                t[s * self.k + c] = dot(qs, cv);
            }
        }
        t
    }
}
