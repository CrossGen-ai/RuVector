//! AVQ + per-vector norm rescaling. Vectors are unit-normalised for
//! codebook training and encoding; a per-vector norm is stored as an
//! `f32` side channel and multiplied into the ADC score at query time.
//!
//! This decouples direction quantization from magnitude, which
//! typically improves recall for inner-product search on heterogeneous
//! magnitude corpora (Guo et al. §5; Malkov & Yashunin note the same
//! effect for HNSW+IVF).

use crate::avq_kmeans::{anisotropic_loss, precompute_directions, train_avq};
use crate::kmeans::lut_ip;
use crate::{AnisotropicConfig, AvqError, Quantizer, QuantizerConfig};

pub struct AvqNorm {
    pub cfg: QuantizerConfig,
    pub aniso: AnisotropicConfig,
    pub dim: usize,
    pub sub_dim: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Per-vector norm side channel, filled at encode time.
    pub norms: Vec<f32>,
}

impl AvqNorm {
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
            norms: Vec::new(),
        })
    }

    fn split<'a>(&self, v: &'a [f32], m_idx: usize) -> &'a [f32] {
        let s = m_idx * self.sub_dim;
        &v[s..s + self.sub_dim]
    }

    fn unit_normalise(v: &[f32]) -> (Vec<f32>, f32) {
        let n2: f32 = v.iter().map(|x| x * x).sum();
        let n = n2.sqrt();
        if n > 0.0 {
            (v.iter().map(|x| x / n).collect(), n)
        } else {
            (v.to_vec(), 0.0)
        }
    }
}

impl Quantizer for AvqNorm {
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
        // Unit-normalise entire training set once.
        let unit: Vec<Vec<f32>> =
            data.iter().map(|v| Self::unit_normalise(v).0).collect();
        let eta = self.aniso.eta(self.sub_dim);
        self.codebooks.clear();
        for m in 0..self.cfg.m {
            let slices: Vec<Vec<f32>> =
                unit.iter().map(|v| self.split(v, m).to_vec()).collect();
            let seed = self.cfg.seed
                ^ (m as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
                ^ 0xB0BA_CAFEu64;
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
        let mut codes = Vec::with_capacity(data.len() * m);
        // AvqNorm stores per-vector norms in `self.norms`.
        // We can't take `&mut self` because trait is `&self`, so we
        // fill into a scratch vector and expect callers to `set_norms`.
        // For the built-in benchmark and tests we expose helpers below.
        for v in data {
            if v.len() != self.dim {
                return Err(AvqError::ShapeMismatch {
                    expected: self.dim,
                    actual: v.len(),
                });
            }
            let (unit, _) = Self::unit_normalise(v);
            for mi in 0..m {
                let sub = self.split(&unit, mi).to_vec();
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
                codes.push(best);
            }
        }
        Ok(codes)
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
            // Rescale by the stored per-vector norm — that recovers the
            // magnitude the encoder threw away.
            let scale = if i < self.norms.len() { self.norms[i] } else { 1.0 };
            scores.push(s * scale);
        }
        Ok(scores)
    }

    fn code_bytes(&self) -> usize {
        self.cfg.m
    }

    fn side_bytes(&self) -> usize {
        4 // f32 per vector
    }
}

impl AvqNorm {
    /// Encode + populate `self.norms` in a single pass. Use this from
    /// applications; the trait-level `encode` deliberately keeps the
    /// side channel out of the return type.
    pub fn encode_with_norms(&mut self, data: &[Vec<f32>]) -> Result<Vec<u8>, AvqError> {
        self.norms.clear();
        for v in data {
            let (_, n) = Self::unit_normalise(v);
            self.norms.push(n);
        }
        self.encode(data)
    }
}
