//! Additive Quantization with beam-search encoding.
//!
//! For M codebooks × K centroids in R^d, a vector `x` is approximated as
//! `x̂ = Σ_m c_m[i_m]`. Encoding uses a beam search of width `beam` over
//! codebooks in a fixed order 0..M: at each step m, every partial encoding
//! `(i_0..i_{m-1})` is extended by every k ∈ [0,K) and the top `beam`
//! survivors — by residual norm `‖x − Σ c_{m'}[i_{m'}]‖²` — are kept.
//! Beam width 1 recovers a purely greedy residual encoder; larger beams
//! approach the exact ILP solution at the cost of encode time.
//!
//! Search uses the AQ-standard ADC expansion under L2:
//!   ‖q − x̂‖² = ‖q‖² − 2 Σ_m q·c_m[i_m] + Σ_m ‖c_m[i_m]‖²
//!               + 2 Σ_{m<m'} c_m[i_m] · c_{m'}[i_{m'}]
//! We precompute per-query `q_dot` (M·K values) once and reuse the static
//! self-norm and cross-codebook inner-product tables built at train time.

use crate::kmeans::kmeans;
use crate::{dot, sq_l2, AqError, Quantizer};

pub struct AqParams {
    pub m: usize,
    pub k: usize,
    pub beam: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Default for AqParams {
    fn default() -> Self {
        Self { m: 4, k: 256, beam: 8, kmeans_iters: 12, seed: 0xBEEF }
    }
}

pub struct AqBeamIndex {
    d: usize,
    m: usize,
    k: usize,
    beam: usize,
    /// codebooks[m][k] = R^d
    codebooks: Vec<Vec<Vec<f32>>>,
    /// self_norm[m][k] = ‖c_m[k]‖²
    self_norm: Vec<Vec<f32>>,
    /// cross[m][m'][k][k'] for m < m' ; stored flat by (m, m'): flat index
    /// = m * M + m' (m < m'). Each entry is a Vec of length K*K.
    cross: Vec<Vec<f32>>, // len M*M ; only entries with m < m' populated.
    name: &'static str,
}

impl AqBeamIndex {
    pub fn train(data: &[Vec<f32>], p: AqParams) -> Result<Self, AqError> {
        if data.is_empty() {
            return Err(AqError::EmptyTraining);
        }
        if p.m == 0 {
            return Err(AqError::InvalidM);
        }
        if p.k == 0 || p.k > 256 || !p.k.is_power_of_two() {
            return Err(AqError::InvalidK);
        }
        let d = data[0].len();

        // Initial codebooks via successive residual k-means (a standard AQ
        // initializer: fit C_0 to X, subtract, fit C_1 to residuals, etc.).
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(p.m);
        let mut residuals: Vec<Vec<f32>> = data.to_vec();
        for m in 0..p.m {
            let cb = kmeans(&residuals, p.k, d, p.kmeans_iters, p.seed ^ m as u64);
            // Subtract nearest centroid to form next residuals.
            for v in residuals.iter_mut() {
                let mut best = 0usize;
                let mut best_d = f32::MAX;
                for (i, c) in cb.iter().enumerate() {
                    let dv = sq_l2(v, c);
                    if dv < best_d {
                        best_d = dv;
                        best = i;
                    }
                }
                let c = &cb[best];
                for j in 0..d {
                    v[j] -= c[j];
                }
            }
            codebooks.push(cb);
        }

        // Precompute self norms and cross tables.
        let mut self_norm: Vec<Vec<f32>> = Vec::with_capacity(p.m);
        for m in 0..p.m {
            self_norm.push(codebooks[m].iter().map(|c| dot(c, c)).collect());
        }
        let mut cross: Vec<Vec<f32>> = vec![Vec::new(); p.m * p.m];
        for m in 0..p.m {
            for mp in (m + 1)..p.m {
                let mut tab = vec![0.0f32; p.k * p.k];
                let cm = &codebooks[m];
                let cmp = &codebooks[mp];
                for i in 0..p.k {
                    for j in 0..p.k {
                        tab[i * p.k + j] = dot(&cm[i], &cmp[j]);
                    }
                }
                cross[m * p.m + mp] = tab;
            }
        }

        let name: &'static str = match p.beam {
            1 => "AQ (beam=1, greedy)",
            8 => "AQ (beam=8)",
            32 => "AQ (beam=32)",
            _ => "AQ (beam=custom)",
        };

        Ok(Self {
            d,
            m: p.m,
            k: p.k,
            beam: p.beam,
            codebooks,
            self_norm,
            cross,
            name,
        })
    }

    /// Beam-search encode a single vector.
    fn encode_one(&self, x: &[f32]) -> Vec<u8> {
        // A beam entry: (code_so_far, residual, cost = ‖residual‖²).
        struct Entry { code: Vec<u8>, residual: Vec<f32>, cost: f32 }
        let mut beam: Vec<Entry> = vec![Entry { code: Vec::new(), residual: x.to_vec(), cost: dot(x, x) }];

        for m in 0..self.m {
            let cb = &self.codebooks[m];
            let mut candidates: Vec<Entry> = Vec::with_capacity(beam.len() * self.k);
            for e in &beam {
                for (k, c) in cb.iter().enumerate() {
                    let mut new_res = e.residual.clone();
                    let mut new_cost = 0.0f32;
                    for j in 0..self.d {
                        new_res[j] -= c[j];
                        new_cost += new_res[j] * new_res[j];
                    }
                    let mut new_code = Vec::with_capacity(e.code.len() + 1);
                    new_code.extend_from_slice(&e.code);
                    new_code.push(k as u8);
                    candidates.push(Entry { code: new_code, residual: new_res, cost: new_cost });
                }
            }
            candidates.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(self.beam);
            beam = candidates;
        }
        beam.into_iter().next().map(|e| e.code).unwrap_or_default()
    }
}

impl Quantizer for AqBeamIndex {
    fn dim(&self) -> usize { self.d }
    fn code_bytes(&self) -> usize { self.m }
    fn name(&self) -> &'static str { self.name }

    fn encode_dataset(&self, xs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        xs.iter().map(|v| self.encode_one(v)).collect()
    }

    fn reconstruct(&self, code: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.d];
        for m in 0..self.m {
            let c = &self.codebooks[m][code[m] as usize];
            for j in 0..self.d {
                out[j] += c[j];
            }
        }
        out
    }

    fn search(&self, query: &[f32], codes: &[Vec<u8>], k: usize) -> Vec<(usize, f32)> {
        // Precompute q_dot[m][k] = q · c_m[k].
        let mut q_dot: Vec<Vec<f32>> = Vec::with_capacity(self.m);
        for m in 0..self.m {
            q_dot.push(self.codebooks[m].iter().map(|c| dot(query, c)).collect());
        }
        let q_norm = dot(query, query);
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(codes.len());
        for (idx, code) in codes.iter().enumerate() {
            let mut linear = 0.0f32;
            let mut self_sum = 0.0f32;
            for m in 0..self.m {
                let km = code[m] as usize;
                linear += q_dot[m][km];
                self_sum += self.self_norm[m][km];
            }
            let mut cross_sum = 0.0f32;
            for m in 0..self.m {
                let km = code[m] as usize;
                for mp in (m + 1)..self.m {
                    let kmp = code[mp] as usize;
                    let tab = &self.cross[m * self.m + mp];
                    cross_sum += tab[km * self.k + kmp];
                }
            }
            let dist = q_norm - 2.0 * linear + self_sum + 2.0 * cross_sum;
            scored.push((idx, dist));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
