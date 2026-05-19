//! Single-level Locally-Adaptive Vector Quantization.
//!
//! For each vector x:
//!   c   = x - mean
//!   lo  = min_i c_i,  hi = max_i c_i
//!   Δ   = (hi - lo) / (2^B - 1)
//!   q_i = round((c_i - lo) / Δ)   ∈ [0, 2^B - 1]
//!   x̂_i = Δ * q_i + lo + mean_i
//!
//! Per-vector overhead: (Δ, lo, ‖x̂‖²). Code: B bits per component, packed.

use crate::distance::{ip, l2_sq, Metric};
use crate::error::LvqError;
use crate::quantizer::{Encoded, Quantizer};

#[derive(Debug, Clone)]
pub struct LvqOne {
    pub dim: usize,
    pub bits: u8,
    pub mean: Vec<f32>,
}

impl LvqOne {
    pub fn new(dim: usize, bits: u8) -> Result<Self, LvqError> {
        if bits != 4 && bits != 8 {
            return Err(LvqError::UnsupportedBits(bits));
        }
        Ok(Self { dim, bits, mean: vec![0.0; dim] })
    }

    #[inline]
    fn levels(&self) -> u32 {
        (1u32 << self.bits) - 1
    }

    fn pack(&self, raw: &[u8]) -> Vec<u8> {
        if self.bits == 8 {
            return raw.to_vec();
        }
        // 4-bit packing: low nibble = even, high nibble = odd.
        let mut out = Vec::with_capacity(self.dim.div_ceil(2));
        let mut i = 0;
        while i < raw.len() {
            let lo = raw[i] & 0x0F;
            let hi = if i + 1 < raw.len() { raw[i + 1] & 0x0F } else { 0 };
            out.push(lo | (hi << 4));
            i += 2;
        }
        out
    }

    fn unpack(&self, code: &[u8], out: &mut Vec<u8>) {
        out.clear();
        if self.bits == 8 {
            out.extend_from_slice(code);
            out.truncate(self.dim);
            return;
        }
        for &b in code {
            out.push(b & 0x0F);
            if out.len() < self.dim {
                out.push((b >> 4) & 0x0F);
            }
        }
        out.truncate(self.dim);
    }
}

impl Quantizer for LvqOne {
    fn name(&self) -> &'static str {
        match self.bits {
            8 => "LVQ1-8",
            4 => "LVQ1-4",
            _ => "LVQ1",
        }
    }
    fn dim(&self) -> usize { self.dim }
    fn bits_per_component(&self) -> u8 { self.bits }
    fn code_bytes(&self) -> usize {
        match self.bits {
            8 => self.dim,
            4 => self.dim.div_ceil(2),
            _ => 0,
        }
    }

    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError> {
        if training.is_empty() { return Err(LvqError::EmptyTraining); }
        let d = self.dim;
        let mut mean = vec![0f32; d];
        for v in training {
            if v.len() != d { return Err(LvqError::DimMismatch { expected: d, got: v.len() }); }
            for i in 0..d { mean[i] += v[i]; }
        }
        let inv = 1.0 / training.len() as f32;
        for m in mean.iter_mut() { *m *= inv; }
        self.mean = mean;
        Ok(())
    }

    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError> {
        if v.len() != self.dim { return Err(LvqError::DimMismatch { expected: self.dim, got: v.len() }); }
        let mut centered = Vec::with_capacity(self.dim);
        for i in 0..self.dim { centered.push(v[i] - self.mean[i]); }

        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &c in &centered {
            if c < lo { lo = c; }
            if c > hi { hi = c; }
        }
        let levels = self.levels() as f32;
        let delta = if hi > lo { (hi - lo) / levels } else { 1.0 };
        let inv_delta = 1.0 / delta;

        let mut raw = Vec::with_capacity(self.dim);
        let mut decoded_sq = 0f32;
        for i in 0..self.dim {
            let q = ((centered[i] - lo) * inv_delta).round();
            let q = q.clamp(0.0, levels) as u8;
            raw.push(q);
            let dec = delta * q as f32 + lo + self.mean[i];
            decoded_sq += dec * dec;
        }
        Ok(Encoded {
            code: self.pack(&raw),
            scale: delta,
            bias: lo,
            decoded_sq_norm: decoded_sq,
            residual: None,
        })
    }

    fn decode(&self, e: &Encoded, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dim);
        let mut buf = Vec::with_capacity(self.dim);
        self.unpack(&e.code, &mut buf);
        for i in 0..self.dim {
            out[i] = e.scale * buf[i] as f32 + e.bias + self.mean[i];
        }
        if let Some(res) = &e.residual {
            // Apply residual correction.
            let mut rbuf = Vec::with_capacity(self.dim);
            // Residual uses the same packing rules at its own bit width.
            let resid_bits = if res.code.len() == self.dim { 8 } else { 4 };
            if resid_bits == 8 {
                for i in 0..self.dim {
                    out[i] += res.scale * res.code[i] as f32 + res.bias;
                }
            } else {
                for &b in &res.code {
                    rbuf.push(b & 0x0F);
                    if rbuf.len() < self.dim { rbuf.push((b >> 4) & 0x0F); }
                }
                rbuf.truncate(self.dim);
                for i in 0..self.dim {
                    out[i] += res.scale * rbuf[i] as f32 + res.bias;
                }
            }
        }
    }

    fn distance(&self, q: &[f32], q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32 {
        // Asymmetric: decode on-the-fly and accumulate ⟨q, x̂⟩.
        // For B=8 we can iterate code directly; for B=4 we unpack 2-at-a-time.
        let dot;
        if self.bits == 8 {
            // x̂_i = scale * code_i + bias + mean_i
            // ⟨q, x̂⟩ = scale * Σ q_i * code_i + bias * Σ q_i + Σ q_i * mean_i
            let mut a = 0f32;
            let mut sum_q = 0f32;
            let mut b = 0f32;
            for i in 0..self.dim {
                a += q[i] * e.code[i] as f32;
                sum_q += q[i];
                b += q[i] * self.mean[i];
            }
            dot = e.scale * a + e.bias * sum_q + b;
        } else {
            let mut a = 0f32;
            let mut sum_q = 0f32;
            let mut b = 0f32;
            let mut idx = 0;
            for &byte in &e.code {
                if idx < self.dim {
                    let v = (byte & 0x0F) as f32;
                    a += q[idx] * v;
                    sum_q += q[idx];
                    b += q[idx] * self.mean[idx];
                    idx += 1;
                }
                if idx < self.dim {
                    let v = ((byte >> 4) & 0x0F) as f32;
                    a += q[idx] * v;
                    sum_q += q[idx];
                    b += q[idx] * self.mean[idx];
                    idx += 1;
                }
            }
            dot = e.scale * a + e.bias * sum_q + b;
        }

        // Residual correction (decoded densely; uncommon path, kept simple).
        if let Some(_res) = &e.residual {
            let mut decoded = vec![0f32; self.dim];
            self.decode(e, &mut decoded);
            return match metric {
                Metric::Ip => -ip(q, &decoded),
                Metric::L2 => l2_sq(q, &decoded),
            };
        }

        match metric {
            Metric::Ip => -dot, // we sort ascending; lower = closer for L2; for IP, max similarity = min(-dot)
            Metric::L2 => q_sq_norm - 2.0 * dot + e.decoded_sq_norm,
        }
    }
}
