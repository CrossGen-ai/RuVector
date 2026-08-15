//! Per-dimension uniform 8-bit scalar quantizer (baseline).

use crate::Quantizer;

/// Per-dimension uniform 8-bit scalar quantizer.
pub struct Sq8 {
    dim: usize,
    /// Per-dimension minimum.
    lo: Vec<f32>,
    /// Per-dimension scale (255 / (max - min)); 0 if the dim is degenerate.
    scale: Vec<f32>,
    /// Inverse scale for decode.
    inv_scale: Vec<f32>,
}

impl Sq8 {
    pub fn train(train: &[f32], dim: usize) -> Self {
        assert!(train.len() % dim == 0 && !train.is_empty());
        let n = train.len() / dim;
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for i in 0..n {
            let x = &train[i * dim..(i + 1) * dim];
            for d in 0..dim {
                if x[d] < lo[d] {
                    lo[d] = x[d];
                }
                if x[d] > hi[d] {
                    hi[d] = x[d];
                }
            }
        }
        let mut scale = vec![0.0f32; dim];
        let mut inv_scale = vec![0.0f32; dim];
        for d in 0..dim {
            let range = hi[d] - lo[d];
            if range > 1e-9 {
                scale[d] = 255.0 / range;
                inv_scale[d] = range / 255.0;
            }
        }
        Self { dim, lo, scale, inv_scale }
    }
}

impl Quantizer for Sq8 {
    fn dim(&self) -> usize {
        self.dim
    }
    fn code_bytes(&self) -> usize {
        self.dim
    }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.dim);
        for d in 0..self.dim {
            let v = ((x[d] - self.lo[d]) * self.scale[d]).round();
            out[d] = v.clamp(0.0, 255.0) as u8;
        }
    }
    fn adc_sq_distance(&self, query: &[f32], encoded: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for d in 0..self.dim {
            let recon = self.lo[d] + encoded[d] as f32 * self.inv_scale[d];
            let diff = query[d] - recon;
            s += diff * diff;
        }
        s
    }
    fn name(&self) -> &'static str {
        "sq8"
    }
}
