//! AVQ score-aware PQ: identical footprint to [`crate::pq::PqMse`], but
//! codebooks are trained by the anisotropic loss so parallel (score)
//! error is penalised more than orthogonal error.

use crate::avq_kmeans::{anisotropic_loss, precompute_directions, train_avq};
use crate::kmeans::lut_ip;
use crate::{AnisotropicConfig, AvqError, Quantizer, QuantizerConfig};

pub struct AvqScoreAware {
    pub cfg: QuantizerConfig,
    pub aniso: AnisotropicConfig,
    pub dim: usize,
    pub sub_dim: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl AvqScoreAware {
    pub fn new(
        cfg: QuantizerConfig,
        aniso: AnisotropicConfig,
        dim: usize,
    ) -> Result<Self, AvqError> {
        if dim % cfg.m != 0 {
            return Err(AvqError::BadSubspaces { dim, m: cfg.m });
        }
        assert!(cfg.ks <= 256, "ks must fit in u8");
        Ok(Self {
            cfg,
            aniso,
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

impl Quantizer for AvqScoreAware {
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
        let eta = self.aniso.eta(self.sub_dim);
        self.codebooks.clear();
        for m in 0..self.cfg.m {
            let slices: Vec<Vec<f32>> =
                data.iter().map(|v| self.split(v, m).to_vec()).collect();
            let seed = self.cfg.seed
                ^ (m as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ 0xA_5_C_A_1_E_D_A_u64;
            let book = train_avq(&slices, self.cfg.ks, self.cfg.iters, eta, seed)?;
            self.codebooks.push(book);
        }
        Ok(())
    }

    fn encode(&self, data: &[Vec<f32>]) -> Result<Vec<u8>, AvqError> {
        if self.codebooks.is_empty() {
            return Err(AvqError::EmptyTraining);
        }
        let eta = self.aniso.eta(self.sub_dim);
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
                let sub = self.split(v, mi).to_vec();
                let (hats, _) = precompute_directions(&[sub.clone()]);
                let mut best = 0u8;
                let mut best_l = f32::INFINITY;
                for (i, c) in self.codebooks[mi].iter().enumerate() {
                    let l = anisotropic_loss(&sub, &hats[0], c, eta);
                    if l < best_l {
                        best_l = l;
                        best = i as u8;
                    }
                }
                out.push(best);
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
