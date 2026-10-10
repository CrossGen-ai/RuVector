//! PRVQ: Product-Residual Vector Quantizer (composite estimator) and
//! the IVF/PQ -> RVQ-refine -> full-verify cascade pipeline.

use crate::{dot, l2sq, Pq8, RvqCascade};

/// Product-Residual Vector Quantizer: PQ for the main code, RVQ on
/// the per-vector PQ-residual for a refinement delta. The combined
/// estimator strictly dominates PQ-ADC because the residual term is
/// additive and well-trained to minimize `||x - pq(x) - rvq_resid(x)||`.
///
/// Memory per vector: m (PQ) + K (RVQ) bytes + 8 scalar bytes
/// (cross-term + partial norm). Query cost per candidate: m + K LUT
/// lookups + 1 cached-float fetch; no vector materialization.
pub struct Prvq {
    pub pq: Pq8,
    pub rvq: RvqCascade,
    /// Per-vector precomputed cross-term `2 * <pq(x), rvq_full(x)>`.
    pub cross: Vec<f32>,
}

impl Prvq {
    pub fn train(
        data: &[f32],
        dim: usize,
        m: usize,
        k_rvq: usize,
        iters: usize,
        seed: u64,
    ) -> Self {
        let n = data.len() / dim;
        let pq = Pq8::train(data, dim, m, iters, seed);
        let sub_dim = dim / m;
        // Residual = x - pq_decode(x)
        let mut residual = vec![0.0f32; n * dim];
        for i in 0..n {
            for sub in 0..m {
                let c = pq.codes[sub * n + i] as usize;
                let cb_off = sub * 256 * sub_dim + c * sub_dim;
                let x_off = i * dim + sub * sub_dim;
                for j in 0..sub_dim {
                    residual[x_off + j] = data[x_off + j] - pq.codebooks[cb_off + j];
                }
            }
        }
        let rvq = RvqCascade::train(&residual, dim, k_rvq, iters, seed.wrapping_add(1));
        // Precompute 2 * <pq(x), rvq_full(x)> per vector.
        let mut cross = vec![0.0f32; n];
        let mut pq_recon = vec![0.0f32; dim];
        let mut rvq_recon = vec![0.0f32; dim];
        for i in 0..n {
            pq_recon.iter_mut().for_each(|v| *v = 0.0);
            rvq_recon.iter_mut().for_each(|v| *v = 0.0);
            for sub in 0..m {
                let c = pq.codes[sub * n + i] as usize;
                let cb_off = sub * 256 * sub_dim + c * sub_dim;
                for j in 0..sub_dim {
                    pq_recon[sub * sub_dim + j] = pq.codebooks[cb_off + j];
                }
            }
            for s in 0..k_rvq {
                let c = rvq.codes[s * n + i];
                let cen = rvq.stages[s].centroid(c);
                for j in 0..dim {
                    rvq_recon[j] += cen[j];
                }
            }
            cross[i] = 2.0 * dot(&pq_recon, &rvq_recon);
        }
        Prvq { pq, rvq, cross }
    }

    /// Estimated l2sq using PQ + RVQ. O(m + K) per candidate.
    /// l2(q, x) ~ ADC_pq + ||rvq_recon||^2 + 2<pq_recon, rvq_recon>
    ///                     - 2<q, rvq_recon>
    pub fn estimate_l2sq(
        &self,
        i: usize,
        pq_lut: &[f32],
        rvq_dot_luts: &[Vec<f32>],
    ) -> f32 {
        let adc = self.pq.adc(i, pq_lut);
        let n = self.rvq.n;
        let k_rvq = self.rvq.stages.len();
        let rvq_norm = self.rvq.partial_norms[(k_rvq - 1) * n + i];
        let mut dot_qr = 0.0f32;
        for s in 0..k_rvq {
            let c = self.rvq.codes[s * n + i] as usize;
            dot_qr += rvq_dot_luts[s][c];
        }
        adc + rvq_norm + self.cross[i] - 2.0 * dot_qr
    }

    pub fn memory_bytes(&self) -> usize {
        self.pq.memory_bytes() + self.rvq.memory_bytes() + self.cross.len() * 4
    }
}

/// Standard IVF cell search → PQ shortlist → RVQ refine → full verify.
pub struct CascadePipeline {
    pub pq: Pq8,
    pub rvq: RvqCascade,
    pub data: Vec<f32>,
    pub dim: usize,
    pub n: usize,
}

impl CascadePipeline {
    /// Run the cascade. `shortlist_pq` is the top-N kept after ADC;
    /// `shortlist_rvq1` after RVQ stage 0; `shortlist_final` gets
    /// full-precision verification.
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        shortlist_pq: usize,
        shortlist_rvq1: usize,
        shortlist_final: usize,
    ) -> Vec<u32> {
        let pq_lut = self.pq.l2_luts(q);
        let mut cand: Vec<(u32, f32)> = (0..self.n as u32)
            .map(|i| (i, self.pq.adc(i as usize, &pq_lut)))
            .collect();
        let keep = shortlist_pq.min(cand.len());
        cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(keep);

        let dot_luts = self.rvq.build_dot_luts(q);
        let qn: f32 = q.iter().map(|&x| x * x).sum();
        for p in cand.iter_mut() {
            p.1 = self.rvq.estimate_l2sq(p.0 as usize, 0, &dot_luts, qn);
        }
        let keep = shortlist_rvq1.min(cand.len());
        cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(keep);

        let last = self.rvq.stages.len() - 1;
        for p in cand.iter_mut() {
            p.1 = self.rvq.estimate_l2sq(p.0 as usize, last, &dot_luts, qn);
        }
        let keep = shortlist_final.min(cand.len());
        cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(keep);

        for p in cand.iter_mut() {
            let v = &self.data[p.0 as usize * self.dim..(p.0 as usize + 1) * self.dim];
            p.1 = l2sq(q, v);
        }
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(k);
        cand.into_iter().map(|x| x.0).collect()
    }
}
