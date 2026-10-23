//! Scalar quantization codecs.
//!
//! * `UniformSQ4`: 4-bit uniform scalar quantization. 2 dims/byte, 8× compression vs f32.
//! * `HadamardSQ4`: randomised Hadamard rotation, then `UniformSQ4` on the rotated vectors.
//! * `HadamardSQ8`: same rotation, 8-bit uniform SQ. 4× compression vs f32.
//!
//! All codecs share the `Codec` trait: `train`, `encode`, `decode`, and
//! `l2_sq_asymmetric` for query-vs-code distance without decoding.

use crate::hadamard::Rotation;

pub trait Codec {
    /// Human-readable name.
    fn name(&self) -> &'static str;
    /// Input dimensionality the codec is trained on.
    fn input_dim(&self) -> usize;
    /// Bits per vector after encoding (including any padding overhead).
    fn bits_per_vector(&self) -> usize;
    /// Encode one vector. Returns a byte slice with `code_bytes()` bytes.
    fn encode(&self, x: &[f32], out: &mut [u8]);
    fn code_bytes(&self) -> usize;
    /// Reconstruct one vector into `out` (same length as `input_dim`).
    fn decode(&self, code: &[u8], out: &mut [f32]);
    /// Compute asymmetric L2^2 distance between a raw query `x` and encoded `code`.
    fn l2_sq_asymmetric(&self, x: &[f32], code: &[u8]) -> f32;
}

/// Uniform 4-bit scalar quantizer with a global min/max per dim-group.
#[derive(Clone, Debug)]
pub struct UniformSQ4 {
    dim: usize,
    lo: f32,
    hi: f32,
    scale: f32,     // (2^4 - 1) / (hi - lo)
    inv_scale: f32, // (hi - lo) / (2^4 - 1)
}

impl UniformSQ4 {
    pub fn train(dim: usize, data: &[f32]) -> Self {
        assert_eq!(data.len() % dim, 0);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in data { if v < lo { lo = v; } if v > hi { hi = v; } }
        if hi <= lo { hi = lo + 1e-6; }
        let scale = 15.0 / (hi - lo);
        Self { dim, lo, hi, scale, inv_scale: 1.0 / scale }
    }

    #[inline]
    fn q(&self, v: f32) -> u8 {
        let z = ((v - self.lo) * self.scale + 0.5).floor();
        z.clamp(0.0, 15.0) as u8
    }

    #[inline]
    fn dq(&self, c: u8) -> f32 { self.lo + (c as f32) * self.inv_scale }
}

impl Codec for UniformSQ4 {
    fn name(&self) -> &'static str { "uniform-sq4" }
    fn input_dim(&self) -> usize { self.dim }
    fn bits_per_vector(&self) -> usize { self.dim * 4 }
    fn code_bytes(&self) -> usize { (self.dim + 1) / 2 }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(out.len(), self.code_bytes());
        for b in out.iter_mut() { *b = 0; }
        for i in 0..self.dim {
            let c = self.q(x[i]);
            if i & 1 == 0 { out[i / 2] = c; } else { out[i / 2] |= c << 4; }
        }
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        for i in 0..self.dim {
            let nib = if i & 1 == 0 { code[i / 2] & 0x0F } else { code[i / 2] >> 4 };
            out[i] = self.dq(nib);
        }
    }
    fn l2_sq_asymmetric(&self, x: &[f32], code: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.dim {
            let nib = if i & 1 == 0 { code[i / 2] & 0x0F } else { code[i / 2] >> 4 };
            let d = x[i] - self.dq(nib);
            s += d * d;
        }
        s
    }
}

/// Uniform 8-bit scalar quantizer with a global min/max.
#[derive(Clone, Debug)]
pub struct UniformSQ8 {
    dim: usize,
    lo: f32,
    scale: f32,
    inv_scale: f32,
}

impl UniformSQ8 {
    pub fn train(dim: usize, data: &[f32]) -> Self {
        assert_eq!(data.len() % dim, 0);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in data { if v < lo { lo = v; } if v > hi { hi = v; } }
        if hi <= lo { hi = lo + 1e-6; }
        let scale = 255.0 / (hi - lo);
        Self { dim, lo, scale, inv_scale: 1.0 / scale }
    }
    #[inline]
    fn q(&self, v: f32) -> u8 {
        let z = ((v - self.lo) * self.scale + 0.5).floor();
        z.clamp(0.0, 255.0) as u8
    }
    #[inline]
    fn dq(&self, c: u8) -> f32 { self.lo + (c as f32) * self.inv_scale }
}

impl Codec for UniformSQ8 {
    fn name(&self) -> &'static str { "uniform-sq8" }
    fn input_dim(&self) -> usize { self.dim }
    fn bits_per_vector(&self) -> usize { self.dim * 8 }
    fn code_bytes(&self) -> usize { self.dim }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        for i in 0..self.dim { out[i] = self.q(x[i]); }
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        for i in 0..self.dim { out[i] = self.dq(code[i]); }
    }
    fn l2_sq_asymmetric(&self, x: &[f32], code: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.dim {
            let d = x[i] - self.dq(code[i]);
            s += d * d;
        }
        s
    }
}

/// Randomised-Hadamard then uniform SQ4.
#[derive(Clone, Debug)]
pub struct HadamardSQ4 {
    rot: Rotation,
    sq: UniformSQ4,
    buf_dim: usize,
}

impl HadamardSQ4 {
    /// Train from an unrotated dataset.
    pub fn train(dim: usize, data: &[f32], seed: u64) -> Self {
        assert_eq!(data.len() % dim, 0);
        let n = data.len() / dim;
        let rot = Rotation::new(dim, seed);
        let rotated = rot.rotate_dataset(data, n);
        let sq = UniformSQ4::train(rot.output_dim(), &rotated);
        let buf_dim = rot.output_dim();
        Self { rot, sq, buf_dim }
    }
    pub fn padded_dim(&self) -> usize { self.buf_dim }
}

impl Codec for HadamardSQ4 {
    fn name(&self) -> &'static str { "hadamard-sq4" }
    fn input_dim(&self) -> usize { self.rot.input_dim() }
    fn bits_per_vector(&self) -> usize { self.buf_dim * 4 }
    fn code_bytes(&self) -> usize { (self.buf_dim + 1) / 2 }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.rot.apply(x, &mut rotated);
        self.sq.encode(&rotated, out);
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        // Decode only returns the rotated reconstruction (same ambient dim). To
        // recover the original coordinates one would invert the rotation; for
        // asymmetric distance we do not need that, and for symmetric distance
        // we compare in rotated space (isometric).
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.sq.decode(code, &mut rotated);
        for i in 0..self.rot.input_dim().min(self.buf_dim) { out[i] = rotated[i]; }
    }
    fn l2_sq_asymmetric(&self, x: &[f32], code: &[u8]) -> f32 {
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.rot.apply(x, &mut rotated);
        self.sq.l2_sq_asymmetric(&rotated, code)
    }
}

/// Randomised-Hadamard then uniform SQ8.
#[derive(Clone, Debug)]
pub struct HadamardSQ8 {
    rot: Rotation,
    sq: UniformSQ8,
    buf_dim: usize,
}

impl HadamardSQ8 {
    pub fn train(dim: usize, data: &[f32], seed: u64) -> Self {
        assert_eq!(data.len() % dim, 0);
        let n = data.len() / dim;
        let rot = Rotation::new(dim, seed);
        let rotated = rot.rotate_dataset(data, n);
        let sq = UniformSQ8::train(rot.output_dim(), &rotated);
        let buf_dim = rot.output_dim();
        Self { rot, sq, buf_dim }
    }
}

impl Codec for HadamardSQ8 {
    fn name(&self) -> &'static str { "hadamard-sq8" }
    fn input_dim(&self) -> usize { self.rot.input_dim() }
    fn bits_per_vector(&self) -> usize { self.buf_dim * 8 }
    fn code_bytes(&self) -> usize { self.buf_dim }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.rot.apply(x, &mut rotated);
        self.sq.encode(&rotated, out);
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.sq.decode(code, &mut rotated);
        for i in 0..self.rot.input_dim().min(self.buf_dim) { out[i] = rotated[i]; }
    }
    fn l2_sq_asymmetric(&self, x: &[f32], code: &[u8]) -> f32 {
        let mut rotated = vec![0.0f32; self.buf_dim];
        self.rot.apply(x, &mut rotated);
        self.sq.l2_sq_asymmetric(&rotated, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::make_skewed_dataset;

    #[test]
    fn roundtrip_bounds_sq4() {
        let d = 32;
        let data = make_skewed_dataset(256, d, 7);
        let sq = UniformSQ4::train(d, &data);
        let mut code = vec![0u8; sq.code_bytes()];
        let mut rec = vec![0.0f32; d];
        for i in 0..256 {
            let x = &data[i * d..(i + 1) * d];
            sq.encode(x, &mut code);
            sq.decode(&code, &mut rec);
            for j in 0..d {
                let err = (x[j] - rec[j]).abs();
                let step = (sq.hi - sq.lo) / 15.0;
                assert!(err <= step * 0.6 + 1e-5, "err {} > {}", err, step);
            }
        }
    }

    #[test]
    fn hadamard_sq4_reduces_mse_on_skewed_data() {
        let d = 64;
        let n = 1024;
        let data = make_skewed_dataset(n, d, 11);
        let base = UniformSQ4::train(d, &data);
        let hada = HadamardSQ4::train(d, &data, 13);
        let mut bcode = vec![0u8; base.code_bytes()];
        let mut hcode = vec![0u8; hada.code_bytes()];
        let (mut mse_b, mut mse_h) = (0.0f64, 0.0f64);
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            base.encode(x, &mut bcode);
            hada.encode(x, &mut hcode);
            mse_b += base.l2_sq_asymmetric(x, &bcode) as f64;
            mse_h += hada.l2_sq_asymmetric(x, &hcode) as f64;
        }
        mse_b /= n as f64;
        mse_h /= n as f64;
        // Hadamard variant must be strictly better on skewed data.
        assert!(mse_h < mse_b * 0.6, "mse_b={} mse_h={}", mse_b, mse_h);
    }
}
