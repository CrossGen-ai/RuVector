//! Per-vector 4-bit scalar quantizer (baseline).
//!
//! For fair comparison, applies the same SFHT rotation as the RaBitQ /
//! Fused variants — SQ4 without rotation is strictly worse when input
//! coordinates are skewed, so rotating first is standard practice.
//!
//! Code layout per vector (bytes): `[f32 min][f32 step][D/2 packed nibbles]`.

use crate::quantizer::{QueryCtx, Quantizer};
use crate::rotation::SignedHadamard;

pub struct Sq4Quant {
    pub dim: usize,
    rot: SignedHadamard,
}

impl Sq4Quant {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            rot: SignedHadamard::new_seeded(dim, seed),
        }
    }
}

impl Quantizer for Sq4Quant {
    fn code_bytes(&self) -> usize {
        8 + self.dim / 2
    }

    fn name(&self) -> &'static str {
        "SQ4-4bit"
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut vr = v.to_vec();
        self.rot.apply(&mut vr);
        let mn = vr.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = vr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step = if mx > mn { (mx - mn) / 15.0 } else { 1.0 };
        let mut out = Vec::with_capacity(self.code_bytes());
        out.extend_from_slice(&mn.to_le_bytes());
        out.extend_from_slice(&step.to_le_bytes());
        let mut packed = vec![0u8; self.dim / 2];
        for i in 0..self.dim {
            let q = (((vr[i] - mn) / step).round().clamp(0.0, 15.0)) as u8;
            if i % 2 == 0 {
                packed[i / 2] |= q;
            } else {
                packed[i / 2] |= q << 4;
            }
        }
        out.extend(packed);
        out
    }

    fn prepare_query(&self, q: &[f32]) -> QueryCtx {
        let mut qr = q.to_vec();
        self.rot.apply(&mut qr);
        let q_norm_sq = qr.iter().map(|x| x * x).sum::<f32>();
        QueryCtx {
            q_rot: qr,
            q_norm_sq,
        }
    }

    fn distance(&self, code: &[u8], ctx: &QueryCtx) -> f32 {
        let mn = f32::from_le_bytes(code[0..4].try_into().unwrap());
        let step = f32::from_le_bytes(code[4..8].try_into().unwrap());
        let packed = &code[8..];
        let mut d = 0.0f32;
        for i in 0..self.dim {
            let q = if i % 2 == 0 {
                packed[i / 2] & 0xF
            } else {
                packed[i / 2] >> 4
            };
            let x = mn + step * q as f32;
            let diff = ctx.q_rot[i] - x;
            d += diff * diff;
        }
        d
    }
}
