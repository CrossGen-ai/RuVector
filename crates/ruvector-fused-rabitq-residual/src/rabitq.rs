//! Pure RaBitQ 1-bit quantizer (baseline).
//!
//! Code layout per vector (bytes): `[f32 norm][ceil(D/8) sign bits]`.
//! Reconstruction: `v̂ = (r / √D) · s`, with `s ∈ {±1}^D`.
//! Approximate squared-L2 distance to a rotated query `q`:
//!   `||q||² + r² − 2·(r/√D)·⟨q, s⟩`.

use crate::quantizer::{QueryCtx, Quantizer};
use crate::rotation::SignedHadamard;

pub struct RabitQuant {
    pub dim: usize,
    rot: SignedHadamard,
}

impl RabitQuant {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            rot: SignedHadamard::new_seeded(dim, seed),
        }
    }
}

impl Quantizer for RabitQuant {
    fn code_bytes(&self) -> usize {
        4 + self.dim / 8
    }

    fn name(&self) -> &'static str {
        "RaBitQ-1bit"
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut vr = v.to_vec();
        self.rot.apply(&mut vr);
        let norm = vr.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut out = Vec::with_capacity(self.code_bytes());
        out.extend_from_slice(&norm.to_le_bytes());
        let mut bits = vec![0u8; self.dim / 8];
        for (i, x) in vr.iter().enumerate() {
            if *x >= 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend(bits);
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
        let r = f32::from_le_bytes(code[0..4].try_into().unwrap());
        let bits = &code[4..];
        let scale = r / (self.dim as f32).sqrt();
        let mut ip = 0.0f32;
        for i in 0..self.dim {
            let s = if (bits[i / 8] >> (i % 8)) & 1 == 1 {
                1.0
            } else {
                -1.0
            };
            ip += ctx.q_rot[i] * s;
        }
        ctx.q_norm_sq + r * r - 2.0 * scale * ip
    }
}
