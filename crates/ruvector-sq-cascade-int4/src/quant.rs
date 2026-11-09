use std::time::Instant;

pub trait Quantizer: Sync {
    fn name(&self) -> &'static str;
    fn train(&mut self, data: &[Vec<f32>]);
    fn encode(&self, v: &[f32]) -> Vec<u8>;
    fn decode(&self, code: &[u8]) -> Vec<f32>;
    fn bytes_per_vector(&self, dim: usize) -> usize;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct QuantStats {
    pub encode_ns_per_vec: f64,
    pub decode_ns_per_vec: f64,
    pub bytes_per_vec: usize,
}

pub struct Fp32Baseline {
    dim: usize,
}

impl Fp32Baseline {
    pub fn new(dim: usize) -> Self { Self { dim } }
}

impl Quantizer for Fp32Baseline {
    fn name(&self) -> &'static str { "fp32" }
    fn train(&mut self, _: &[Vec<f32>]) {}
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
        out
    }
    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let d = code.len() / 4;
        let mut out = Vec::with_capacity(d);
        for i in 0..d {
            let b: [u8; 4] = code[i*4..i*4+4].try_into().unwrap();
            out.push(f32::from_le_bytes(b));
        }
        out
    }
    fn bytes_per_vector(&self, dim: usize) -> usize { dim * 4 }
}

pub struct Int8Sq {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>, // per-dim scale so q = round((x-mean)/scale), q in [-127,127]
}

impl Int8Sq {
    pub fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], scale: vec![1.0; dim] }
    }
}

impl Quantizer for Int8Sq {
    fn name(&self) -> &'static str { "int8_sq" }
    fn train(&mut self, data: &[Vec<f32>]) {
        let d = self.mean.len();
        let n = data.len() as f32;
        // per-dim mean
        for row in data {
            for j in 0..d { self.mean[j] += row[j]; }
        }
        for j in 0..d { self.mean[j] /= n; }
        // per-dim max-abs after centering, then scale = max / 127
        let mut maxabs = vec![0.0f32; d];
        for row in data {
            for j in 0..d {
                let c = (row[j] - self.mean[j]).abs();
                if c > maxabs[j] { maxabs[j] = c; }
            }
        }
        for j in 0..d {
            self.scale[j] = (maxabs[j] / 127.0).max(1e-8);
        }
    }
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let d = v.len();
        let mut out = vec![0u8; d];
        for j in 0..d {
            let q = ((v[j] - self.mean[j]) / self.scale[j]).round().clamp(-127.0, 127.0) as i8;
            out[j] = q as u8;
        }
        out
    }
    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let d = code.len();
        let mut out = vec![0.0f32; d];
        for j in 0..d {
            let q = code[j] as i8 as f32;
            out[j] = self.mean[j] + q * self.scale[j];
        }
        out
    }
    fn bytes_per_vector(&self, dim: usize) -> usize { dim }
}

pub struct CascadeInt8Int4 {
    base: Int8Sq,
    res_scale: Vec<f32>, // per-dim residual scale; q4 in [-7,7]
}

impl CascadeInt8Int4 {
    pub fn new(dim: usize) -> Self {
        Self { base: Int8Sq::new(dim), res_scale: vec![1.0; dim] }
    }
}

impl Quantizer for CascadeInt8Int4 {
    fn name(&self) -> &'static str { "cascade_int8_int4" }
    fn train(&mut self, data: &[Vec<f32>]) {
        self.base.train(data);
        let d = self.base.mean.len();
        // measure residual max-abs after base reconstruction
        let mut maxabs = vec![0.0f32; d];
        for row in data {
            let c = self.base.encode(row);
            let r = self.base.decode(&c);
            for j in 0..d {
                let e = (row[j] - r[j]).abs();
                if e > maxabs[j] { maxabs[j] = e; }
            }
        }
        for j in 0..d {
            // signed int4: range [-7,7], use 7 as max magnitude.
            self.res_scale[j] = (maxabs[j] / 7.0).max(1e-8);
        }
    }
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let d = v.len();
        let base = self.base.encode(v);
        let recon = self.base.decode(&base);
        // packed int4 residual: 2 nibbles per byte
        let packed_len = (d + 1) / 2;
        let mut packed = vec![0u8; packed_len];
        for j in 0..d {
            let r = v[j] - recon[j];
            let q = (r / self.res_scale[j]).round().clamp(-7.0, 7.0) as i8;
            // store 4-bit two's complement in nibble: map [-7..7] -> [1..15] with 0 reserved, but use mask
            let nib = (q as u8) & 0x0F;
            if j & 1 == 0 {
                packed[j / 2] = (packed[j / 2] & 0xF0) | nib;
            } else {
                packed[j / 2] = (packed[j / 2] & 0x0F) | (nib << 4);
            }
        }
        let mut out = base;
        out.extend_from_slice(&packed);
        out
    }
    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let d = self.base.mean.len();
        let base = &code[..d];
        let packed = &code[d..];
        let recon = self.base.decode(base);
        let mut out = vec![0.0f32; d];
        for j in 0..d {
            let byte = packed[j / 2];
            let nib = if j & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            // sign-extend 4-bit
            let q = if nib & 0x08 != 0 { (nib | 0xF0) as i8 } else { nib as i8 };
            out[j] = recon[j] + (q as f32) * self.res_scale[j];
        }
        out
    }
    fn bytes_per_vector(&self, dim: usize) -> usize { dim + (dim + 1) / 2 }
}

pub fn measure<Q: Quantizer>(q: &Q, data: &[Vec<f32>]) -> QuantStats {
    let dim = data[0].len();
    let iters = data.len().min(20_000);
    let t = Instant::now();
    let mut codes = Vec::with_capacity(iters);
    for i in 0..iters { codes.push(q.encode(&data[i])); }
    let enc_ns = (t.elapsed().as_nanos() as f64) / (iters as f64);
    let t = Instant::now();
    let mut sink = 0.0f32;
    for c in &codes { sink += q.decode(c)[0]; }
    std::hint::black_box(sink);
    let dec_ns = (t.elapsed().as_nanos() as f64) / (iters as f64);
    QuantStats {
        encode_ns_per_vec: enc_ns,
        decode_ns_per_vec: dec_ns,
        bytes_per_vec: q.bytes_per_vector(dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::gaussian_dataset;

    fn mse(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len();
        let mut s = 0.0f64;
        for i in 0..n { let d = (a[i] - b[i]) as f64; s += d * d; }
        s / (n as f64)
    }

    #[test]
    fn cascade_beats_int8_in_mse() {
        let data = gaussian_dataset(5000, 64, 42);
        let mut int8 = Int8Sq::new(64);
        int8.train(&data);
        let mut casc = CascadeInt8Int4::new(64);
        casc.train(&data);
        let mut err8 = 0.0f64;
        let mut errc = 0.0f64;
        for v in &data[..500] {
            err8 += mse(v, &int8.decode(&int8.encode(v)));
            errc += mse(v, &casc.decode(&casc.encode(v)));
        }
        println!("int8 mse={:.6e}  cascade mse={:.6e}", err8, errc);
        assert!(errc < err8, "cascade MSE {} should be < int8 MSE {}", errc, err8);
    }

    #[test]
    fn cascade_storage_is_1_5x_int8() {
        let c = CascadeInt8Int4::new(128);
        let i8 = Int8Sq::new(128);
        assert_eq!(c.bytes_per_vector(128), 128 + 64);
        assert_eq!(i8.bytes_per_vector(128), 128);
    }

    #[test]
    fn roundtrip_preserves_shape() {
        let data = gaussian_dataset(100, 32, 7);
        let mut q = CascadeInt8Int4::new(32);
        q.train(&data);
        let c = q.encode(&data[0]);
        let d = q.decode(&c);
        assert_eq!(d.len(), 32);
    }
}
