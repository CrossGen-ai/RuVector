//! Baseline product quantization trained by plain MSE k-means.

use crate::kmeans::{assign, lut_ip, train_mse};
use crate::{AvqError, Quantizer, QuantizerConfig};

/// Baseline PQ: per-subspace codebooks trained by MSE. `ks` is capped
/// at 256 so each subspace code is exactly one byte.
pub struct PqMse {
    pub cfg: QuantizerConfig,
    pub dim: usize,
    pub sub_dim: usize,
    /// `M` codebooks, each `Ks × sub_dim`.
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl PqMse {
    pub fn new(cfg: QuantizerConfig, dim: usize) -> Result<Self, AvqError> {
        if dim % cfg.m != 0 {
            return Err(AvqError::BadSubspaces { dim, m: cfg.m });
        }
        assert!(cfg.ks <= 256, "ks must fit in u8");
        Ok(Self {
            cfg,
            dim,
            sub_dim: dim / cfg.m,
            codebooks: Vec::new(),
        })
    }

    pub fn split<'a>(&self, v: &'a [f32], m_idx: usize) -> &'a [f32] {
        let s = m_idx * self.sub_dim;
        &v[s..s + self.sub_dim]
    }
}

impl Quantizer for PqMse {
    fn train(&mut self, data: &[Vec<f32>]) -> Result<(), AvqError> {
        if data.is_empty() {
            return Err(AvqError::EmptyTraining);
        }
        if data[0].len() != self.dim {
            return Err(AvqError::ShapeMismatch {
                expected: self.dim,
                actual: data[0].len(),
            });
        }
        self.codebooks.clear();
        for m in 0..self.cfg.m {
            let slices: Vec<Vec<f32>> =
                data.iter().map(|v| self.split(v, m).to_vec()).collect();
            let seed = self.cfg.seed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let book = train_mse(&slices, self.cfg.ks, self.cfg.iters, seed)?;
            self.codebooks.push(book);
        }
        Ok(())
    }

    fn encode(&self, data: &[Vec<f32>]) -> Result<Vec<u8>, AvqError> {
        if self.codebooks.is_empty() {
            return Err(AvqError::EmptyTraining);
        }
        let m = self.cfg.m;
        let mut out = Vec::with_capacity(data.len() * m);
        for v in data {
            if v.len() != self.dim {
                return Err(AvqError::ShapeMismatch {
                    expected: self.dim,
                    actual: v.len(),
                });
            }
            for mi in 0..m {
                let s = self.split(v, mi);
                let a = assign(s, &self.codebooks[mi]);
                out.push(a as u8);
            }
        }
        Ok(out)
    }

    fn adc(&self, query: &[f32], codes: &[u8]) -> Result<Vec<f32>, AvqError> {
        if self.codebooks.is_empty() {
            return Err(AvqError::EmptyTraining);
        }
        if query.len() != self.dim {
            return Err(AvqError::ShapeMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let m = self.cfg.m;
        let ks = self.cfg.ks;
        // Build LUTs: [M][Ks] inner products of query slice · centroid.
        let mut lut: Vec<Vec<f32>> = Vec::with_capacity(m);
        for mi in 0..m {
            lut.push(lut_ip(self.split(query, mi), &self.codebooks[mi]));
        }
        let n = codes.len() / m;
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * m;
            let mut s = 0.0;
            for mi in 0..m {
                let code = codes[base + mi] as usize;
                debug_assert!(code < ks);
                s += lut[mi][code];
            }
            scores.push(s);
        }
        Ok(scores)
    }

    fn code_bytes(&self) -> usize {
        self.cfg.m
    }
}
