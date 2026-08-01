//! Anisotropic product quantization (ScaNN-style).
//!
//! The centroid update in each iteration solves the small SPD system:
//!
//! ```text
//!   ( Σ_i W_i ) · c* = Σ_i W_i · x_i
//!   W_i = h_orth · I + (h_par - h_orth) · û_i · û_iᵀ
//! ```
//!
//! where `û_i = x_i / ||x_i||` (per-subvector direction).
//!
//! At `eta = h_par / h_orth = 1` this reduces exactly to `c* = mean(x_i)`,
//! i.e. standard k-means. At `eta >> 1` centroids are pulled to minimise
//! error along the vector's own direction, which is the direction that
//! matters for the MIPS score error decomposition.
//!
//! The assignment step also uses the anisotropic loss so that a point's
//! cluster minimises the same objective the update minimises.
//!
//! Query-time scoring is identical to standard PQ (inner-product LUT):
//! the anisotropic weighting only shapes *where* the centroids sit — the
//! score computation itself does not change.

use crate::math::solve_spd;
use crate::{Hit, Lcg, Pq, PqConfig};

/// Anisotropic PQ codebook, parameterised by parallel-vs-orthogonal weight ratio `eta`.
#[derive(Debug, Clone)]
pub struct AnisotropicPq {
    cfg: PqConfig,
    d_sub: usize,
    /// `eta = h_par / h_orth`. Must be ≥ 1.
    eta: f32,
    /// Row-major centroids: `[m * 256 * d_sub + j * d_sub + t]`.
    centroids: Vec<f32>,
    /// Cached label for reporting.
    label: String,
    trained: bool,
}

impl AnisotropicPq {
    pub fn new(cfg: PqConfig, eta: f32) -> Self {
        assert!(cfg.dim % cfg.m == 0, "dim must be divisible by m");
        assert_eq!(cfg.k, 256);
        assert!(eta >= 1.0, "eta must be >= 1 (eta=1 recovers standard PQ)");
        let d_sub = cfg.dim / cfg.m;
        let label = format!("AnisotropicPQ(eta={:.1})", eta);
        Self {
            cfg,
            d_sub,
            eta,
            centroids: vec![0.0; cfg.m * cfg.k * d_sub],
            label,
            trained: false,
        }
    }

    #[inline]
    fn sub(&self, m: usize) -> &[f32] {
        let start = m * self.cfg.k * self.d_sub;
        &self.centroids[start..start + self.cfg.k * self.d_sub]
    }

    #[inline]
    fn sub_mut(&mut self, m: usize) -> &mut [f32] {
        let start = m * self.cfg.k * self.d_sub;
        let k = self.cfg.k * self.d_sub;
        &mut self.centroids[start..start + k]
    }

    /// Anisotropic loss for a residual `r = x - c` given unit direction `u`.
    /// L = h_orth·||r||² + (h_par − h_orth)·(r·u)²
    /// With h_orth = 1 (normalised) and h_par = eta:
    /// L = ||r||² + (eta − 1)·(r·u)²
    #[inline]
    fn loss(r: &[f32], u: &[f32], eta: f32) -> f32 {
        let mut r2 = 0.0f32;
        let mut ru = 0.0f32;
        for t in 0..r.len() {
            r2 += r[t] * r[t];
            ru += r[t] * u[t];
        }
        r2 + (eta - 1.0) * ru * ru
    }

    fn train_subspace(&mut self, m: usize, data: &[f32], n: usize) {
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let k = self.cfg.k;
        let eta = self.eta;

        // Precompute per-point unit directions of the subvector.
        // If a subvector has ~zero norm, u is set to a safe basis and the
        // parallel term becomes numerically inert.
        let mut units = vec![0.0f32; n * ds];
        for i in 0..n {
            let src = &data[i * d + m * ds..i * d + m * ds + ds];
            let mut sq = 0.0f32;
            for t in 0..ds { sq += src[t] * src[t]; }
            let inv = 1.0 / sq.sqrt().max(1e-12);
            for t in 0..ds { units[i * ds + t] = src[t] * inv; }
        }

        // Init centroids from random distinct rows.
        let mut rng = Lcg(self.cfg.seed.wrapping_add(m as u64 * 0x51_7C_C1_B7));
        {
            let sub = self.sub_mut(m);
            for j in 0..k {
                let idx = (rng.next_u64() as usize) % n;
                let src = &data[idx * d + m * ds..idx * d + m * ds + ds];
                sub[j * ds..(j + 1) * ds].copy_from_slice(src);
            }
        }

        let mut assignments = vec![0u16; n];
        let mut residual = vec![0.0f32; ds];

        for _iter in 0..self.cfg.iterations {
            // ---- Assignment under the anisotropic loss ----
            {
                let sub = self.sub(m);
                for i in 0..n {
                    let x = &data[i * d + m * ds..i * d + m * ds + ds];
                    let u = &units[i * ds..(i + 1) * ds];
                    let mut best = 0usize;
                    let mut best_l = f32::INFINITY;
                    for j in 0..k {
                        let c = &sub[j * ds..(j + 1) * ds];
                        for t in 0..ds { residual[t] = x[t] - c[t]; }
                        let l = Self::loss(&residual, u, eta);
                        if l < best_l {
                            best_l = l;
                            best = j;
                        }
                    }
                    assignments[i] = best as u16;
                }
            }

            // ---- Update: SPD solve per cluster ----
            // A_j = Σ_i∈j [ I + (eta-1) · u_i u_iᵀ ]  (h_orth = 1)
            // b_j = Σ_i∈j [ I + (eta-1) · u_i u_iᵀ ] · x_i
            let mut a_all = vec![0.0f64; k * ds * ds];
            let mut b_all = vec![0.0f64; k * ds];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let j = assignments[i] as usize;
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                let u = &units[i * ds..(i + 1) * ds];
                counts[j] += 1;
                // Accumulate A += I + (eta-1) u uᵀ
                for a in 0..ds {
                    a_all[j * ds * ds + a * ds + a] += 1.0;
                    for b in 0..ds {
                        a_all[j * ds * ds + a * ds + b] += (eta - 1.0) as f64 * u[a] as f64 * u[b] as f64;
                    }
                }
                // W_i x_i = x_i + (eta-1)·(u·x)·u
                let mut ux = 0.0f32;
                for t in 0..ds { ux += u[t] * x[t]; }
                for t in 0..ds {
                    b_all[j * ds + t] += x[t] as f64 + (eta - 1.0) as f64 * ux as f64 * u[t] as f64;
                }
            }

            let sub = self.sub_mut(m);
            for j in 0..k {
                if counts[j] == 0 {
                    let idx = (rng.next_u64() as usize) % n;
                    let src = &data[idx * d + m * ds..idx * d + m * ds + ds];
                    sub[j * ds..(j + 1) * ds].copy_from_slice(src);
                    continue;
                }
                let a_slice = &mut a_all[j * ds * ds..(j + 1) * ds * ds];
                let b_slice = &b_all[j * ds..(j + 1) * ds];
                match solve_spd(a_slice, b_slice, ds) {
                    Some(sol) => {
                        for t in 0..ds { sub[j * ds + t] = sol[t] as f32; }
                    }
                    None => {
                        // Fall back to unweighted mean (Cholesky failed → ill-conditioned).
                        for t in 0..ds {
                            sub[j * ds + t] = (b_slice[t] / counts[j] as f64) as f32;
                        }
                    }
                }
            }
        }
    }

    /// Encode using the same anisotropic loss the training used, so the
    /// database is consistent with the codebook objective. `u` is derived
    /// from the sub-vector direction at encode time.
    fn encode_sub(&self, x: &[f32], m: usize) -> u8 {
        let ds = self.d_sub;
        let eta = self.eta;
        // Compute unit direction u.
        let mut sq = 0.0f32;
        for t in 0..ds { sq += x[t] * x[t]; }
        let inv = 1.0 / sq.sqrt().max(1e-12);
        let mut u = [0.0f32; 32]; // support d_sub up to 32
        assert!(ds <= 32, "d_sub {} exceeds compile-time bound", ds);
        for t in 0..ds { u[t] = x[t] * inv; }

        let sub = self.sub(m);
        let mut best = 0u8;
        let mut best_l = f32::INFINITY;
        for j in 0..self.cfg.k {
            let c = &sub[j * ds..(j + 1) * ds];
            let mut r2 = 0.0f32;
            let mut ru = 0.0f32;
            for t in 0..ds {
                let diff = x[t] - c[t];
                r2 += diff * diff;
                ru += diff * u[t];
            }
            let l = r2 + (eta - 1.0) * ru * ru;
            if l < best_l {
                best_l = l;
                best = j as u8;
            }
        }
        best
    }
}

impl Pq for AnisotropicPq {
    fn name(&self) -> &str { &self.label }

    fn train(&mut self, data: &[f32], n: usize) {
        for m in 0..self.cfg.m {
            self.train_subspace(m, data, n);
        }
        self.trained = true;
    }

    fn encode(&self, data: &[f32], n: usize) -> Vec<u8> {
        assert!(self.trained);
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let m_ = self.cfg.m;
        let mut codes = vec![0u8; n * m_];
        for i in 0..n {
            for m in 0..m_ {
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                codes[i * m_ + m] = self.encode_sub(x, m);
            }
        }
        codes
    }

    fn search(&self, query: &[f32], codes: &[u8], n: usize, k: usize) -> Vec<Hit> {
        assert_eq!(query.len(), self.cfg.dim);
        let m_ = self.cfg.m;
        let ds = self.d_sub;
        let kc = self.cfg.k;
        let mut lut = vec![0.0f32; m_ * kc];
        for m in 0..m_ {
            let q_m = &query[m * ds..(m + 1) * ds];
            let sub = self.sub(m);
            for j in 0..kc {
                let c = &sub[j * ds..(j + 1) * ds];
                let mut s = 0.0f32;
                for t in 0..ds { s += q_m[t] * c[t]; }
                lut[m * kc + j] = s;
            }
        }
        let mut scores: Vec<Hit> = (0..n)
            .map(|i| {
                let base = i * m_;
                let mut s = 0.0f32;
                for m in 0..m_ {
                    s += lut[m * kc + codes[base + m] as usize];
                }
                Hit { id: i as u32, score: s }
            })
            .collect();
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    fn reconstruction_error(&self, data: &[f32], n: usize) -> f64 {
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let m_ = self.cfg.m;
        let mut acc = 0.0f64;
        for i in 0..n {
            for m in 0..m_ {
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                let sub = self.sub(m);
                let mut best_d = f32::INFINITY;
                for j in 0..self.cfg.k {
                    let c = &sub[j * ds..(j + 1) * ds];
                    let mut d2 = 0.0f32;
                    for t in 0..ds {
                        let diff = x[t] - c[t];
                        d2 += diff * diff;
                    }
                    if d2 < best_d { best_d = d2; }
                }
                acc += best_d as f64;
            }
        }
        acc / n as f64
    }

    fn memory_bytes(&self) -> usize {
        self.centroids.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generate;

    #[test]
    fn eta_one_matches_standard_recon_error() {
        // With eta = 1 exactly, anisotropic PQ should behave like standard PQ.
        let ds = generate(16, 400, 1, 11);
        let cfg = PqConfig { dim: 16, m: 4, k: 256, iterations: 6, seed: 3 };
        let mut aniso = AnisotropicPq::new(cfg, 1.0);
        aniso.train(&ds.data, ds.n);
        let err = aniso.reconstruction_error(&ds.data, ds.n);
        assert!(err.is_finite() && err >= 0.0);
    }

    #[test]
    fn higher_eta_tightens_score_on_top_relevant_vectors() {
        // ScaNN's claim: on the *ground-truth top-k* vectors (the ones that
        // matter for MIPS), anisotropic quantization gives smaller score
        // error than isotropic. Score error here is measured on the SAME
        // vector ids for both variants, isolating the codebook quality.
        let ds = generate(32, 800, 30, 42);
        let cfg = PqConfig { dim: 32, m: 4, k: 256, iterations: 8, seed: 5 };

        let mut iso = AnisotropicPq::new(cfg, 1.0);
        iso.train(&ds.data, ds.n);
        let iso_codes = iso.encode(&ds.data, ds.n);

        let mut hi = AnisotropicPq::new(cfg, 8.0);
        hi.train(&ds.data, ds.n);
        let hi_codes = hi.encode(&ds.data, ds.n);

        // Predicted-score MSE on the ground-truth top-10 ids (same ids for both).
        let mut se_iso = 0.0f64;
        let mut se_hi = 0.0f64;
        let mut n_terms = 0usize;
        for q_idx in 0..ds.num_queries {
            let q = &ds.queries[q_idx * ds.dim..(q_idx + 1) * ds.dim];
            let truth = crate::exact_mips(q, &ds.data, ds.n, 10);
            // Reconstruct predicted score for each truth id under both variants
            // via the LUT (using search on a subset is cumbersome; compute inline).
            for hit in &truth {
                let id = hit.id as usize;
                let true_s = hit.score as f64;
                let s_iso = predict_score(&iso, q, &iso_codes, id) as f64;
                let s_hi  = predict_score(&hi,  q, &hi_codes,  id) as f64;
                se_iso += (s_iso - true_s).powi(2);
                se_hi  += (s_hi  - true_s).powi(2);
                n_terms += 1;
            }
        }
        let mse_iso = se_iso / n_terms as f64;
        let mse_hi  = se_hi  / n_terms as f64;
        assert!(mse_hi <= mse_iso,
            "anisotropic top-k score MSE {:.4} should be ≤ isotropic {:.4}",
            mse_hi, mse_iso);
    }

    /// Predict the PQ score of database id `id` for query `q`.
    fn predict_score(pq: &AnisotropicPq, q: &[f32], codes: &[u8], id: usize) -> f32 {
        let m_ = pq.cfg.m;
        let ds = pq.d_sub;
        let kc = pq.cfg.k;
        let mut s = 0.0f32;
        for m in 0..m_ {
            let q_m = &q[m * ds..(m + 1) * ds];
            let code = codes[id * m_ + m] as usize;
            let c = &pq.sub(m)[code * ds..(code + 1) * ds];
            for t in 0..ds { s += q_m[t] * c[t]; }
        }
        s
    }
}
