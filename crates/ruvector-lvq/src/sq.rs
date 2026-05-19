//! Global 8-bit scalar quantization baseline.
//!
//! Fits a single (lo, hi) per dimension across the training set, quantizes to
//! 8-bit codes. No per-vector adaptation. This is the classic SQ8 baseline
//! that LVQ is designed to outperform.

use crate::distance::{ip, l2_sq, Metric};
use crate::error::LvqError;
use crate::quantizer::{Encoded, Quantizer};

#[derive(Debug, Clone)]
pub struct Sq8 {
    pub dim: usize,
    pub lo: Vec<f32>,
    pub delta: Vec<f32>,
}

impl Sq8 {
    pub fn new(dim: usize) -> Self {
        Self { dim, lo: vec![0.0; dim], delta: vec![1.0; dim] }
    }
}

impl Quantizer for Sq8 {
    fn name(&self) -> &'static str { "SQ8" }
    fn dim(&self) -> usize { self.dim }
    fn bits_per_component(&self) -> u8 { 8 }
    fn code_bytes(&self) -> usize { self.dim }

    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError> {
        if training.is_empty() { return Err(LvqError::EmptyTraining); }
        let d = self.dim;
        let mut lo = vec![f32::INFINITY; d];
        let mut hi = vec![f32::NEG_INFINITY; d];
        for v in training {
            if v.len() != d { return Err(LvqError::DimMismatch { expected: d, got: v.len() }); }
            for i in 0..d {
                if v[i] < lo[i] { lo[i] = v[i]; }
                if v[i] > hi[i] { hi[i] = v[i]; }
            }
        }
        let mut delta = vec![1.0f32; d];
        for i in 0..d {
            delta[i] = if hi[i] > lo[i] { (hi[i] - lo[i]) / 255.0 } else { 1.0 };
        }
        self.lo = lo;
        self.delta = delta;
        Ok(())
    }

    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError> {
        if v.len() != self.dim { return Err(LvqError::DimMismatch { expected: self.dim, got: v.len() }); }
        let mut code = Vec::with_capacity(self.dim);
        let mut decoded_sq = 0f32;
        for i in 0..self.dim {
            let q = ((v[i] - self.lo[i]) / self.delta[i]).round().clamp(0.0, 255.0) as u8;
            code.push(q);
            let dec = self.delta[i] * q as f32 + self.lo[i];
            decoded_sq += dec * dec;
        }
        Ok(Encoded { code, scale: 0.0, bias: 0.0, decoded_sq_norm: decoded_sq, residual: None })
    }

    fn decode(&self, e: &Encoded, out: &mut [f32]) {
        for i in 0..self.dim {
            out[i] = self.delta[i] * e.code[i] as f32 + self.lo[i];
        }
    }

    fn distance(&self, q: &[f32], _q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32 {
        let mut decoded = vec![0f32; self.dim];
        self.decode(e, &mut decoded);
        match metric {
            Metric::L2 => l2_sq(q, &decoded),
            Metric::Ip => -ip(q, &decoded),
        }
    }
}
