//! Scalar quantizers (trait-based), fit per cell/head/tail segment.

/// Per-dimension min/max used by linear scalar quantization.
#[derive(Clone, Debug)]
pub struct SqParams {
    pub scale: Vec<f32>,  // step = (max-min)/(levels-1) per dim
    pub offset: Vec<f32>, // min per dim
}

impl SqParams {
    pub fn zero(dim: usize) -> Self {
        Self {
            scale: vec![1.0; dim],
            offset: vec![0.0; dim],
        }
    }
    /// Fit per-dim min/max from a slice of vectors (each &[f32]).
    pub fn fit(vectors: &[&[f32]], dim: usize) -> Self {
        if vectors.is_empty() {
            return Self::zero(dim);
        }
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for v in vectors {
            for d in 0..dim {
                let x = v[d];
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        for d in 0..dim {
            if maxs[d] <= mins[d] {
                maxs[d] = mins[d] + 1e-6;
            }
        }
        Self {
            scale: (0..dim).map(|d| maxs[d] - mins[d]).collect(),
            offset: mins,
        }
    }
}

pub trait Quantizer {
    /// bits per dim, used for packing.
    const BITS: u8;
    /// Number of quantization levels.
    const LEVELS: u32;

    fn encode_many(vectors: &[&[f32]], p: &SqParams) -> Vec<u8>;
    fn decode_into(codes: &[u8], idx: usize, p: &SqParams, out: &mut [f32]);
}

pub struct Sq8;

impl Quantizer for Sq8 {
    const BITS: u8 = 8;
    const LEVELS: u32 = 256;

    fn encode_many(vectors: &[&[f32]], p: &SqParams) -> Vec<u8> {
        if vectors.is_empty() {
            return vec![];
        }
        let dim = vectors[0].len();
        let mut out = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            for d in 0..dim {
                let u = ((v[d] - p.offset[d]) / p.scale[d]).clamp(0.0, 1.0);
                let q = (u * 255.0 + 0.5) as u32;
                out.push(q.min(255) as u8);
            }
        }
        out
    }

    fn decode_into(codes: &[u8], idx: usize, p: &SqParams, out: &mut [f32]) {
        let dim = out.len();
        let base = idx * dim;
        for d in 0..dim {
            let q = codes[base + d] as f32;
            out[d] = p.offset[d] + (q / 255.0) * p.scale[d];
        }
    }
}

pub struct Sq4;

impl Quantizer for Sq4 {
    const BITS: u8 = 4;
    const LEVELS: u32 = 16;

    /// Encode each vector into (dim + 1) / 2 bytes: two 4-bit nibbles per byte.
    /// Layout: low nibble = dim d (even), high nibble = dim d (odd).
    fn encode_many(vectors: &[&[f32]], p: &SqParams) -> Vec<u8> {
        if vectors.is_empty() {
            return vec![];
        }
        let dim = vectors[0].len();
        let bytes_per = (dim + 1) / 2;
        let mut out = vec![0u8; vectors.len() * bytes_per];
        for (vi, v) in vectors.iter().enumerate() {
            let base = vi * bytes_per;
            for d in 0..dim {
                let u = ((v[d] - p.offset[d]) / p.scale[d]).clamp(0.0, 1.0);
                let q = (u * 15.0 + 0.5) as u32;
                let q = q.min(15) as u8;
                let byte_i = base + d / 2;
                if d % 2 == 0 {
                    out[byte_i] |= q & 0x0F;
                } else {
                    out[byte_i] |= (q & 0x0F) << 4;
                }
            }
        }
        out
    }

    fn decode_into(codes: &[u8], idx: usize, p: &SqParams, out: &mut [f32]) {
        let dim = out.len();
        let bytes_per = (dim + 1) / 2;
        let base = idx * bytes_per;
        for d in 0..dim {
            let byte_i = base + d / 2;
            let q = if d % 2 == 0 {
                codes[byte_i] & 0x0F
            } else {
                (codes[byte_i] >> 4) & 0x0F
            };
            out[d] = p.offset[d] + (q as f32 / 15.0) * p.scale[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_roundtrip_within_bound() {
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let vs = vec![v.as_slice()];
        let p = SqParams::fit(&vs, 32);
        let codes = Sq8::encode_many(&vs, &p);
        let mut out = vec![0.0f32; 32];
        Sq8::decode_into(&codes, 0, &p, &mut out);
        for i in 0..32 {
            let err = (v[i] - out[i]).abs();
            let step = p.scale[i] / 255.0;
            assert!(err <= step + 1e-6, "sq8 err {err} > step {step}");
        }
    }

    #[test]
    fn sq4_roundtrip_within_bound() {
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let vs = vec![v.as_slice()];
        let p = SqParams::fit(&vs, 32);
        let codes = Sq4::encode_many(&vs, &p);
        let mut out = vec![0.0f32; 32];
        Sq4::decode_into(&codes, 0, &p, &mut out);
        for i in 0..32 {
            let err = (v[i] - out[i]).abs();
            let step = p.scale[i] / 15.0;
            assert!(err <= step + 1e-6, "sq4 err {err} > step {step}");
        }
    }
}
