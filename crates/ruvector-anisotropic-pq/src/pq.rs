//! Product Quantizer with anisotropic training and ADC search.

use rayon::prelude::*;

use crate::kmeans::{train_subspace, SubLossWeights};

#[derive(Copy, Clone, Debug)]
pub enum LossKind {
    /// Standard PQ: minimize ||x - x_hat||^2.
    Reconstruction,
    /// ScaNN-style: weight parallel residual by `eta` (typical 2..8).
    Anisotropic { eta: f32 },
    /// Anisotropic with per-example eta interpolated between
    /// eta_min and eta_max based on ||x|| percentile within batch.
    LearnedNorm { eta_min: f32, eta_max: f32 },
}

#[derive(Clone, Debug)]
pub struct PqConfig {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub iters: usize,
    pub loss: LossKind,
    pub seed: u64,
}

pub struct AnisotropicPq {
    pub d: usize,
    pub m: usize,
    pub sub_d: usize,
    pub k: usize,
    /// Codebooks: `m` blocks of `k * sub_d` f32.
    pub codebooks: Vec<Vec<f32>>,
    pub loss: LossKind,
}

impl AnisotropicPq {
    pub fn train(cfg: &PqConfig, data: &[f32]) -> Self {
        assert!(cfg.d % cfg.m == 0, "d must be divisible by m");
        let sub_d = cfg.d / cfg.m;
        let n = data.len() / cfg.d;
        assert!(n >= cfg.k, "need n>=k training points");

        // Precompute per-example weights and parent-direction restrictions.
        let (weights, parent_dir) = compute_directions(cfg, data, sub_d);

        // Train each subspace in parallel.
        let seed = cfg.seed;
        let codebooks: Vec<Vec<f32>> = (0..cfg.m)
            .into_par_iter()
            .map(|m_idx| {
                let mut sub = Vec::with_capacity(n * sub_d);
                let mut pd = Vec::with_capacity(n * sub_d);
                for i in 0..n {
                    let base = i * cfg.d + m_idx * sub_d;
                    sub.extend_from_slice(&data[base..base + sub_d]);
                    let pdb = i * cfg.d + m_idx * sub_d;
                    pd.extend_from_slice(&parent_dir[pdb..pdb + sub_d]);
                }
                train_subspace(&sub, &pd, &weights, sub_d, cfg.k, cfg.iters, seed.wrapping_add(m_idx as u64 * 7919))
            })
            .collect();

        Self {
            d: cfg.d,
            m: cfg.m,
            sub_d,
            k: cfg.k,
            codebooks,
            loss: cfg.loss,
        }
    }

    /// Encode a batch of full-d vectors. Output layout: n * m bytes (u8 for K<=256, else u16).
    pub fn encode_batch(&self, data: &[f32]) -> Vec<u16> {
        let n = data.len() / self.d;
        let mut out = vec![0u16; n * self.m];
        out.par_chunks_mut(self.m).enumerate().for_each(|(i, row)| {
            for m in 0..self.m {
                let base = i * self.d + m * self.sub_d;
                let sub = &data[base..base + self.sub_d];
                row[m] = self.assign_sub(m, sub) as u16;
            }
        });
        out
    }

    fn assign_sub(&self, m: usize, sub: &[f32]) -> usize {
        let cb = &self.codebooks[m];
        let mut best = f32::MAX;
        let mut best_c = 0usize;
        for c in 0..self.k {
            let cent = &cb[c * self.sub_d..(c + 1) * self.sub_d];
            let mut s = 0f32;
            for j in 0..self.sub_d {
                let d = sub[j] - cent[j];
                s += d * d;
            }
            if s < best {
                best = s;
                best_c = c;
            }
        }
        best_c
    }

    /// Build the query ADC lookup table: m * k entries. For MIPS/cosine we
    /// use `-⟨q_sub, c⟩` so that argmin over sum = argmax inner product.
    /// When `dist_l2 = true`, entries are ||q_sub - c||^2 (standard PQ L2 ADC).
    pub fn build_lut(&self, q: &[f32], dist_l2: bool) -> Vec<f32> {
        assert_eq!(q.len(), self.d);
        let mut lut = vec![0f32; self.m * self.k];
        for m in 0..self.m {
            let qs = &q[m * self.sub_d..(m + 1) * self.sub_d];
            let cb = &self.codebooks[m];
            for c in 0..self.k {
                let cent = &cb[c * self.sub_d..(c + 1) * self.sub_d];
                let mut s = 0f32;
                if dist_l2 {
                    for j in 0..self.sub_d {
                        let d = qs[j] - cent[j];
                        s += d * d;
                    }
                } else {
                    // For MIPS-max: we want scores maximized. Store negative
                    // dot so we can still take argmin.
                    for j in 0..self.sub_d {
                        s -= qs[j] * cent[j];
                    }
                }
                lut[m * self.k + c] = s;
            }
        }
        lut
    }

    /// Score a codeword against the LUT: sum lookups over M subspaces.
    #[inline]
    pub fn score_code(&self, lut: &[f32], code: &[u16]) -> f32 {
        let mut s = 0f32;
        for m in 0..self.m {
            s += lut[m * self.k + code[m] as usize];
        }
        s
    }

    /// Search top-k over an encoded corpus. Returns sorted (score, idx),
    /// smallest score first (best under argmin semantics).
    pub fn search(&self, lut: &[f32], codes: &[u16], top_k: usize) -> Vec<(f32, u32)> {
        let n = codes.len() / self.m;
        // Simple heap-free approach: collect + partial sort.
        let mut scored: Vec<(f32, u32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let c = &codes[i * self.m..(i + 1) * self.m];
                (self.score_code(lut, c), i as u32)
            })
            .collect();
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored
    }

    /// Bytes per vector at rest (log2(k) rounded up per subspace × m).
    pub fn bytes_per_vector(&self) -> usize {
        let bits_per_sub = (self.k as f32).log2().ceil() as usize;
        (self.m * bits_per_sub + 7) / 8
    }
}

fn compute_directions(cfg: &PqConfig, data: &[f32], sub_d: usize) -> (Vec<SubLossWeights>, Vec<f32>) {
    let n = data.len() / cfg.d;
    let mut parent_dir = vec![0f32; n * cfg.d];
    let mut norms = vec![0f32; n];
    for i in 0..n {
        let x = &data[i * cfg.d..(i + 1) * cfg.d];
        let s: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        norms[i] = s;
        for j in 0..cfg.d {
            parent_dir[i * cfg.d + j] = x[j] / s;
        }
    }
    // Renormalize per subspace so `parent_dir` restricted to a subspace is
    // approximately unit-norm — this is the projection used by weighted_loss.
    for i in 0..n {
        for m in 0..cfg.m {
            let base = i * cfg.d + m * sub_d;
            let s: f32 = parent_dir[base..base + sub_d].iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for j in 0..sub_d {
                parent_dir[base + j] /= s;
            }
        }
    }
    let weights: Vec<SubLossWeights> = match cfg.loss {
        LossKind::Reconstruction => vec![SubLossWeights::reconstruction(); n],
        LossKind::Anisotropic { eta } => vec![SubLossWeights { w_parallel: eta, w_perp: 1.0 }; n],
        LossKind::LearnedNorm { eta_min, eta_max } => {
            // Percentile-scale eta by norm.
            let mut sorted = norms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = sorted[sorted.len() / 20];
            let hi = sorted[sorted.len() - sorted.len() / 20 - 1];
            let span = (hi - lo).max(1e-12);
            norms
                .iter()
                .map(|&n| {
                    let t = ((n - lo) / span).clamp(0.0, 1.0);
                    let eta = eta_min + (eta_max - eta_min) * t;
                    SubLossWeights { w_parallel: eta, w_perp: 1.0 }
                })
                .collect()
        }
    };
    (weights, parent_dir)
}

#[derive(Copy, Clone, Debug)]
pub enum Metric { L2, Mips }

/// Recall@top_k evaluator supporting L2 or MIPS ground-truth and matching
/// PQ ADC mode.
pub fn eval_recall_metric(pq: &AnisotropicPq, data: &[f32], codes: &[u16], queries: &[f32], top_k: usize, metric: Metric) -> f32 {
    let nq = queries.len() / pq.d;
    let n = data.len() / pq.d;
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * pq.d..(qi + 1) * pq.d];
        // Exact top_k under the requested metric.
        let mut exact: Vec<(f32, u32)> = (0..n)
            .map(|i| {
                let x = &data[i * pq.d..(i + 1) * pq.d];
                let s = match metric {
                    Metric::L2 => {
                        let mut s = 0f32;
                        for j in 0..pq.d { let d = q[j] - x[j]; s += d * d; }
                        s
                    }
                    Metric::Mips => {
                        // negative dot to keep argmin semantics
                        let mut s = 0f32;
                        for j in 0..pq.d { s -= q[j] * x[j]; }
                        s
                    }
                };
                (s, i as u32)
            })
            .collect();
        exact.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        exact.truncate(top_k);
        let truth: std::collections::HashSet<u32> = exact.iter().map(|(_, i)| *i).collect();

        let dist_l2 = matches!(metric, Metric::L2);
        let lut = pq.build_lut(q, dist_l2);
        let approx = pq.search(&lut, codes, top_k);
        for (_, i) in approx {
            if truth.contains(&i) { hits += 1; }
        }
    }
    hits as f32 / (nq * top_k) as f32
}

/// Backwards-compatible L2 recall wrapper.
pub fn eval_recall(pq: &AnisotropicPq, data: &[f32], codes: &[u16], queries: &[f32], top_k: usize) -> f32 {
    eval_recall_metric(pq, data, codes, queries, top_k, Metric::L2)
}

/// Decompose the reconstruction residual r = x - x_hat into components
/// parallel and perpendicular to x, and return (mse_par, mse_perp).
/// This is the quantity anisotropic training explicitly trades off.
pub fn mse_decomposition(pq: &AnisotropicPq, data: &[f32], codes: &[u16]) -> (f32, f32) {
    let n = codes.len() / pq.m;
    let mut par_sum = 0f64;
    let mut perp_sum = 0f64;
    for i in 0..n {
        let x = &data[i * pq.d..(i + 1) * pq.d];
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let inv = 1.0 / norm;
        // Reconstruct x_hat
        let code = &codes[i * pq.m..(i + 1) * pq.m];
        let mut xhat = vec![0f32; pq.d];
        for m in 0..pq.m {
            let c = code[m] as usize;
            let cent = &pq.codebooks[m][c * pq.sub_d..(c + 1) * pq.sub_d];
            xhat[m * pq.sub_d..(m + 1) * pq.sub_d].copy_from_slice(cent);
        }
        // r = x - xhat; dot with x/|x|
        let mut rd = 0f32;
        for j in 0..pq.d { rd += (x[j] - xhat[j]) * x[j] * inv; }
        for j in 0..pq.d {
            let r = x[j] - xhat[j];
            let par = rd * x[j] * inv;
            let per = r - par;
            par_sum += (par * par) as f64;
            perp_sum += (per * per) as f64;
        }
    }
    ((par_sum / n as f64) as f32, (perp_sum / n as f64) as f32)
}

#[allow(dead_code)]
fn _eval_recall_unused(pq: &AnisotropicPq, data: &[f32], codes: &[u16], queries: &[f32], top_k: usize) -> f32 {
    let nq = queries.len() / pq.d;
    let n = data.len() / pq.d;
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * pq.d..(qi + 1) * pq.d];
        // Exact L2 top_k
        let mut exact: Vec<(f32, u32)> = (0..n)
            .map(|i| {
                let x = &data[i * pq.d..(i + 1) * pq.d];
                let mut s = 0f32;
                for j in 0..pq.d {
                    let d = q[j] - x[j];
                    s += d * d;
                }
                (s, i as u32)
            })
            .collect();
        exact.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        exact.truncate(top_k);
        let truth: std::collections::HashSet<u32> = exact.iter().map(|(_, i)| *i).collect();

        // PQ approx
        let lut = pq.build_lut(q, true);
        let approx = pq.search(&lut, codes, top_k);
        for (_, i) in approx {
            if truth.contains(&i) {
                hits += 1;
            }
        }
    }
    hits as f32 / (nq * top_k) as f32
}
