//! Fused RaBitQ + 4-bit scalar residual quantizer.
//!
//! Stage 1 (RaBitQ): rotate `v` via SFHT, take `s = sign(v_rot)` and
//! record `r = ||v_rot||`. The stage-1 reconstruction is
//! `v̂₁ = (r/√D) · s`.
//!
//! Stage 2 (residual SQ4): compute `e = v_rot − v̂₁`. Because SFHT
//! spreads energy near-isotropically and the RaBitQ direction captures
//! the dominant sign structure, `||e||` is empirically much smaller than
//! `||v_rot||`, so 4 bits per dimension resolve `e` to a fraction of the
//! per-coordinate scale. Encode `e` with per-vector min/max SQ4.
//!
//! Final reconstruction: `v̂ = (r/√D)·s + e_dequant`.
//!
//! Bit budget:
//!   * signs: 1 bit / dim
//!   * residual: 4 bits / dim
//!   * meta:   3·f32 = 96 bits per vector (norm, res_min, res_step)
//!   * total ≈ 5 bits / dim + 96 bits (amortized to <0.2 bits/dim at
//!     D=512).
//!
//! Code layout per vector (bytes):
//! `[f32 norm][f32 res_min][f32 res_step][D/8 sign bits][D/2 packed 4-bit residual]`.

use crate::quantizer::{QueryCtx, Quantizer};
use crate::rotation::SignedHadamard;

pub struct FusedRQR {
    pub dim: usize,
    rot: SignedHadamard,
}

impl FusedRQR {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            dim,
            rot: SignedHadamard::new_seeded(dim, seed),
        }
    }
}

impl Quantizer for FusedRQR {
    fn code_bytes(&self) -> usize {
        12 + self.dim / 8 + self.dim / 2
    }

    fn name(&self) -> &'static str {
        "Fused-RaBitQ+SQ4-Residual"
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut vr = v.to_vec();
        self.rot.apply(&mut vr);
        let norm = vr.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = norm / (self.dim as f32).sqrt();

        let mut signs = vec![0u8; self.dim / 8];
        let mut residual = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let s = if vr[i] >= 0.0 { 1.0 } else { -1.0 };
            if s > 0.0 {
                signs[i / 8] |= 1 << (i % 8);
            }
            residual[i] = vr[i] - scale * s;
        }
        let mn = residual.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = residual.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step = if mx > mn { (mx - mn) / 15.0 } else { 1.0 };
        let mut packed = vec![0u8; self.dim / 2];
        for i in 0..self.dim {
            let q = (((residual[i] - mn) / step).round().clamp(0.0, 15.0)) as u8;
            if i % 2 == 0 {
                packed[i / 2] |= q;
            } else {
                packed[i / 2] |= q << 4;
            }
        }

        let mut out = Vec::with_capacity(self.code_bytes());
        out.extend_from_slice(&norm.to_le_bytes());
        out.extend_from_slice(&mn.to_le_bytes());
        out.extend_from_slice(&step.to_le_bytes());
        out.extend(signs);
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
        let r = f32::from_le_bytes(code[0..4].try_into().unwrap());
        let mn = f32::from_le_bytes(code[4..8].try_into().unwrap());
        let step = f32::from_le_bytes(code[8..12].try_into().unwrap());
        let sign_bytes = self.dim / 8;
        let signs = &code[12..12 + sign_bytes];
        let packed = &code[12 + sign_bytes..];
        let scale = r / (self.dim as f32).sqrt();
        let mut d = 0.0f32;
        for i in 0..self.dim {
            let s = if (signs[i / 8] >> (i % 8)) & 1 == 1 {
                1.0
            } else {
                -1.0
            };
            let q = if i % 2 == 0 {
                packed[i / 2] & 0xF
            } else {
                packed[i / 2] >> 4
            };
            let residual = mn + step * q as f32;
            let x = scale * s + residual;
            let diff = ctx.q_rot[i] - x;
            d += diff * diff;
        }
        d
    }
}
