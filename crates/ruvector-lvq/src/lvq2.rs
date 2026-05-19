//! Two-level LVQ: primary B1-bit code + residual B2-bit code.
//!
//! Residual = (x - decode(primary)). Centering for the residual stage uses
//! its own per-vector min so the second quantizer can use its full range.

use crate::distance::Metric;
use crate::error::LvqError;
use crate::lvq1::LvqOne;
use crate::quantizer::{Encoded, Quantizer};

#[derive(Debug, Clone)]
pub struct LvqTwo {
    pub primary: LvqOne,
    pub residual_bits: u8,
    pub dim: usize,
}

impl LvqTwo {
    pub fn new(dim: usize, primary_bits: u8, residual_bits: u8) -> Result<Self, LvqError> {
        if residual_bits != 4 && residual_bits != 8 {
            return Err(LvqError::UnsupportedBits(residual_bits));
        }
        Ok(Self {
            primary: LvqOne::new(dim, primary_bits)?,
            residual_bits,
            dim,
        })
    }
}

impl Quantizer for LvqTwo {
    fn name(&self) -> &'static str {
        match (self.primary.bits, self.residual_bits) {
            (8, 8) => "LVQ2-8x8",
            (8, 4) => "LVQ2-8x4",
            (4, 4) => "LVQ2-4x4",
            _ => "LVQ2",
        }
    }
    fn dim(&self) -> usize { self.dim }
    fn bits_per_component(&self) -> u8 { self.primary.bits + self.residual_bits }
    fn code_bytes(&self) -> usize {
        let prim = match self.primary.bits {
            8 => self.dim,
            4 => self.dim.div_ceil(2),
            _ => 0,
        };
        let res = match self.residual_bits {
            8 => self.dim,
            4 => self.dim.div_ceil(2),
            _ => 0,
        };
        prim + res + 8 // residual carries its own scale + bias (8 bytes extra)
    }

    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError> {
        self.primary.fit(training)
    }

    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError> {
        let mut primary = self.primary.encode(v)?;

        // Compute residual = v - decode(primary).
        let mut decoded = vec![0f32; self.dim];
        self.primary.decode(&primary, &mut decoded);
        let mut resid = Vec::with_capacity(self.dim);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..self.dim {
            let r = v[i] - decoded[i];
            if r < lo { lo = r; }
            if r > hi { hi = r; }
            resid.push(r);
        }
        let levels = ((1u32 << self.residual_bits) - 1) as f32;
        let delta = if hi > lo { (hi - lo) / levels } else { 1.0 };
        let inv = 1.0 / delta;

        let mut raw = Vec::with_capacity(self.dim);
        let mut decoded_sq = 0f32;
        for i in 0..self.dim {
            let q = ((resid[i] - lo) * inv).round().clamp(0.0, levels) as u8;
            raw.push(q);
            let total = decoded[i] + delta * q as f32 + lo;
            decoded_sq += total * total;
        }

        let res_code = if self.residual_bits == 8 {
            raw
        } else {
            let mut packed = Vec::with_capacity(self.dim.div_ceil(2));
            let mut i = 0;
            while i < raw.len() {
                let l = raw[i] & 0x0F;
                let h = if i + 1 < raw.len() { raw[i + 1] & 0x0F } else { 0 };
                packed.push(l | (h << 4));
                i += 2;
            }
            packed
        };

        primary.decoded_sq_norm = decoded_sq;
        primary.residual = Some(Box::new(Encoded {
            code: res_code,
            scale: delta,
            bias: lo,
            decoded_sq_norm: 0.0,
            residual: None,
        }));
        Ok(primary)
    }

    fn decode(&self, e: &Encoded, out: &mut [f32]) {
        self.primary.decode(e, out);
    }

    fn distance(&self, q: &[f32], q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32 {
        self.primary.distance(q, q_sq_norm, e, metric)
    }
}
