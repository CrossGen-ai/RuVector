//! Two-level residual Product Quantizer + amortised scorer.

use crate::pq::Pq;
use crate::rng::Rng;
use crate::Quantizer;

/// Two-level residual product quantizer.
///
/// Encoding: `code1 = PQ1.encode(x)`; `r = x - PQ1.reconstruct(code1)`;
/// `code2 = PQ2.encode(r)`. Code layout is `[code1 || code2]`.
///
/// Query-time distance uses the identity
/// `||q - (r1 + r2)||^2 = ||(q - r1) - r2||^2`. We form `q' = q - r1`
/// (dependent on `code1`) and evaluate `PQ2.adc(q', code2)`. See
/// [`Rpq2Scorer`] for the standard bucket-by-`code1` amortisation.
pub struct Rpq2 {
    pub(crate) dim: usize,
    pub(crate) pq1: Pq,
    pub(crate) pq2: Pq,
}

impl Rpq2 {
    /// Train an RPQ2: first PQ1 on the raw data, then PQ2 on the residuals.
    pub fn train(train: &[f32], dim: usize, m1: usize, m2: usize, iters: usize, rng: &mut Rng) -> Self {
        let pq1 = Pq::train(train, dim, m1, iters, rng);
        // Build residuals: r_i = train_i - reconstruct(encode(train_i)).
        let n = train.len() / dim;
        let mut residuals = vec![0.0f32; n * dim];
        let mut code = vec![0u8; m1];
        let mut recon = vec![0.0f32; dim];
        for i in 0..n {
            let x = &train[i * dim..(i + 1) * dim];
            pq1.encode_raw(x, &mut code);
            pq1.reconstruct(&code, &mut recon);
            let dst = &mut residuals[i * dim..(i + 1) * dim];
            for d in 0..dim {
                dst[d] = x[d] - recon[d];
            }
        }
        let pq2 = Pq::train(&residuals, dim, m2, iters, rng);
        Self { dim, pq1, pq2 }
    }

    /// Bytes used by the coarse layer.
    pub fn m1(&self) -> usize {
        self.pq1.m()
    }
    /// Bytes used by the residual layer.
    pub fn m2(&self) -> usize {
        self.pq2.m()
    }
    /// Coarse layer (for advanced integration).
    pub fn pq1(&self) -> &Pq {
        &self.pq1
    }
    /// Residual layer.
    pub fn pq2(&self) -> &Pq {
        &self.pq2
    }
}

impl Quantizer for Rpq2 {
    fn dim(&self) -> usize {
        self.dim
    }
    fn code_bytes(&self) -> usize {
        self.pq1.m() + self.pq2.m()
    }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        let m1 = self.pq1.m();
        let (c1, c2) = out.split_at_mut(m1);
        self.pq1.encode_raw(x, c1);
        let mut recon = vec![0.0f32; self.dim];
        self.pq1.reconstruct(c1, &mut recon);
        let mut r = vec![0.0f32; self.dim];
        for d in 0..self.dim {
            r[d] = x[d] - recon[d];
        }
        self.pq2.encode_raw(&r, c2);
    }

    fn adc_sq_distance(&self, query: &[f32], encoded: &[u8]) -> f32 {
        let m1 = self.pq1.m();
        let (c1, c2) = encoded.split_at(m1);
        let mut r1 = vec![0.0f32; self.dim];
        self.pq1.reconstruct(c1, &mut r1);
        let mut qprime = vec![0.0f32; self.dim];
        for d in 0..self.dim {
            qprime[d] = query[d] - r1[d];
        }
        let lut2 = self.pq2.compute_lut(&qprime);
        self.pq2.adc_from_lut(&lut2, c2)
    }
    fn name(&self) -> &'static str {
        "rpq2"
    }
}

/// Amortised scorer for RPQ2 that groups database vectors by their coarse
/// code, so the residual LUT is built once per coarse code rather than
/// once per database vector.
pub struct Rpq2Scorer<'a> {
    rpq: &'a Rpq2,
    buckets: std::collections::BTreeMap<Vec<u8>, Vec<u32>>,
    codes: &'a [u8],
    code_bytes: usize,
}

impl<'a> Rpq2Scorer<'a> {
    pub fn new(rpq: &'a Rpq2, codes: &'a [u8]) -> Self {
        let code_bytes = rpq.code_bytes();
        assert!(!codes.is_empty() && codes.len() % code_bytes == 0);
        let n = codes.len() / code_bytes;
        let m1 = rpq.m1();
        let mut buckets: std::collections::BTreeMap<Vec<u8>, Vec<u32>> = Default::default();
        for i in 0..n {
            let c1 = &codes[i * code_bytes..i * code_bytes + m1];
            buckets.entry(c1.to_vec()).or_default().push(i as u32);
        }
        Self { rpq, buckets, codes, code_bytes }
    }

    /// Score every database vector against `query`, writing squared L2 into `out`.
    pub fn score_all(&self, query: &[f32], out: &mut [f32]) {
        let dim = self.rpq.dim();
        let m1 = self.rpq.m1();
        let m2 = self.rpq.m2();
        let mut r1 = vec![0.0f32; dim];
        let mut qprime = vec![0.0f32; dim];
        for (c1, items) in &self.buckets {
            self.rpq.pq1.reconstruct(c1, &mut r1);
            for d in 0..dim {
                qprime[d] = query[d] - r1[d];
            }
            let lut2 = self.rpq.pq2.compute_lut(&qprime);
            for &i in items {
                let base = i as usize * self.code_bytes + m1;
                let c2 = &self.codes[base..base + m2];
                out[i as usize] = self.rpq.pq2.adc_from_lut(&lut2, c2);
            }
        }
    }
}
