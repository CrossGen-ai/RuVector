//! Baselines: uniform SQ8 and SQ4 for apples-to-apples comparison.

use crate::{describe_dims, FasqError, Quantizer};

/// Uniform per-dim 8-bit scalar quantizer (max-min range calibration).
#[derive(Debug, Clone)]
pub struct UniformSq8 {
    pub dim: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // (max - min) / 255
}

impl UniformSq8 {
    pub fn train(train: &[Vec<f32>]) -> Result<Self, FasqError> {
        let stats = describe_dims(train)?;
        let dim = stats.len();
        let mut mins = vec![0.0; dim];
        let mut scales = vec![0.0; dim];
        for i in 0..dim {
            mins[i] = stats[i].min;
            let r = stats[i].max - stats[i].min;
            scales[i] = if r > 0.0 { r / 255.0 } else { 0.0 };
        }
        Ok(Self { dim, mins, scales })
    }
}

impl Quantizer for UniformSq8 {
    fn bits_per_vector(&self) -> usize { self.dim * 8 }
    fn encode(&self, v: &[f32], out: &mut Vec<u8>) -> Result<usize, FasqError> {
        if v.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: v.len() });
        }
        let start = out.len();
        for i in 0..self.dim {
            let x = if self.scales[i] == 0.0 { 0.0 } else {
                ((v[i] - self.mins[i]) / self.scales[i]).round().clamp(0.0, 255.0)
            };
            out.push(x as u8);
        }
        Ok(out.len() - start)
    }
    fn decode(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), FasqError> {
        if bytes.len() < self.dim || out.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: bytes.len().min(out.len()) });
        }
        for i in 0..self.dim {
            out[i] = self.mins[i] + (bytes[i] as f32) * self.scales[i];
        }
        Ok(())
    }
}

/// Uniform per-dim 4-bit scalar quantizer (nibble-packed, two dims per byte).
#[derive(Debug, Clone)]
pub struct UniformSq4 {
    pub dim: usize,
    pub mins: Vec<f32>,
    pub scales: Vec<f32>, // (max - min) / 15
}

impl UniformSq4 {
    pub fn train(train: &[Vec<f32>]) -> Result<Self, FasqError> {
        let stats = describe_dims(train)?;
        let dim = stats.len();
        let mut mins = vec![0.0; dim];
        let mut scales = vec![0.0; dim];
        for i in 0..dim {
            mins[i] = stats[i].min;
            let r = stats[i].max - stats[i].min;
            scales[i] = if r > 0.0 { r / 15.0 } else { 0.0 };
        }
        Ok(Self { dim, mins, scales })
    }
}

impl Quantizer for UniformSq4 {
    fn bits_per_vector(&self) -> usize { self.dim * 4 }
    fn encode(&self, v: &[f32], out: &mut Vec<u8>) -> Result<usize, FasqError> {
        if v.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: v.len() });
        }
        let nb = (self.dim + 1) / 2;
        let start = out.len();
        out.resize(start + nb, 0);
        let buf = &mut out[start..];
        for i in 0..self.dim {
            let q = if self.scales[i] == 0.0 { 0u8 } else {
                ((v[i] - self.mins[i]) / self.scales[i]).round().clamp(0.0, 15.0) as u8
            };
            let bi = i / 2;
            if i % 2 == 0 {
                buf[bi] |= q << 4;
            } else {
                buf[bi] |= q & 0x0F;
            }
        }
        Ok(nb)
    }
    fn decode(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), FasqError> {
        if out.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: out.len() });
        }
        let nb = (self.dim + 1) / 2;
        if bytes.len() < nb {
            return Err(FasqError::DimMismatch { expected: nb, got: bytes.len() });
        }
        for i in 0..self.dim {
            let bi = i / 2;
            let q = if i % 2 == 0 { (bytes[bi] >> 4) & 0x0F } else { bytes[bi] & 0x0F };
            out[i] = self.mins[i] + (q as f32) * self.scales[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth() -> Vec<Vec<f32>> {
        (0..200).map(|i| (0..8).map(|j| ((i as f32) * 0.1 + j as f32).sin()).collect()).collect()
    }

    #[test]
    fn sq8_roundtrip_within_step() {
        let t = synth();
        let q = UniformSq8::train(&t).unwrap();
        let mut code = Vec::new();
        let mut recon = vec![0.0; 8];
        for v in &t {
            code.clear();
            q.encode(v, &mut code).unwrap();
            q.decode(&code, &mut recon).unwrap();
            for i in 0..8 {
                assert!((v[i] - recon[i]).abs() <= q.scales[i] + 1e-4);
            }
        }
    }

    #[test]
    fn sq4_roundtrip_within_step() {
        let t = synth();
        let q = UniformSq4::train(&t).unwrap();
        let mut code = Vec::new();
        let mut recon = vec![0.0; 8];
        for v in &t {
            code.clear();
            q.encode(v, &mut code).unwrap();
            q.decode(&code, &mut recon).unwrap();
            for i in 0..8 {
                assert!((v[i] - recon[i]).abs() <= q.scales[i] + 1e-4);
            }
        }
    }
}
