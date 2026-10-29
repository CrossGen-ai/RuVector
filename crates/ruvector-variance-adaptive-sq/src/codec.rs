//! Bit-packed VA-SQ codec.

use crate::alloc::{allocate_bits, AllocPolicy};

#[derive(Clone, Debug)]
pub struct VaSqConfig {
    pub dim: usize,
    pub budget_bits_per_dim: u32,
    pub b_min: u32,
    pub b_max: u32,
    pub policy: AllocPolicy,
}

#[derive(Clone, Debug)]
pub struct VaSqCodec {
    pub cfg: VaSqConfig,
    pub bits: Vec<u32>,
    pub mins: Vec<f32>,
    pub steps: Vec<f32>,
    pub bit_offsets: Vec<usize>,
    pub bytes_per_vec: usize,
}

impl VaSqCodec {
    /// Train the codec from a sample of vectors: compute per-dim variance,
    /// allocate bits, and record each dim's min/step for min-max quant.
    pub fn train(sample: &[f32], cfg: VaSqConfig) -> Self {
        let d = cfg.dim;
        assert_eq!(sample.len() % d, 0, "sample length must be a multiple of dim");
        let n = sample.len() / d;
        assert!(n > 1, "need at least two vectors to train");

        let mut mean = vec![0.0_f64; d];
        for i in 0..n {
            for j in 0..d {
                mean[j] += sample[i * d + j] as f64;
            }
        }
        for j in 0..d {
            mean[j] /= n as f64;
        }
        let mut var = vec![0.0_f32; d];
        let mut mins = vec![f32::INFINITY; d];
        let mut maxs = vec![f32::NEG_INFINITY; d];
        for i in 0..n {
            for j in 0..d {
                let x = sample[i * d + j];
                let diff = x as f64 - mean[j];
                var[j] += (diff * diff) as f32;
                if x < mins[j] {
                    mins[j] = x;
                }
                if x > maxs[j] {
                    maxs[j] = x;
                }
            }
        }
        for j in 0..d {
            var[j] /= (n - 1) as f32;
        }

        let bits = allocate_bits(&var, cfg.budget_bits_per_dim, cfg.b_min, cfg.b_max, cfg.policy);

        let mut steps = vec![0.0_f32; d];
        for j in 0..d {
            let levels = (1u32 << bits[j]) as f32;
            let range = (maxs[j] - mins[j]).max(1e-12);
            steps[j] = range / (levels - 1.0).max(1.0);
        }

        let mut bit_offsets = Vec::with_capacity(d + 1);
        let mut off = 0usize;
        bit_offsets.push(0);
        for j in 0..d {
            off += bits[j] as usize;
            bit_offsets.push(off);
        }
        let total_bits = off;
        let bytes_per_vec = (total_bits + 7) / 8;

        Self {
            cfg,
            bits,
            mins,
            steps,
            bit_offsets,
            bytes_per_vec,
        }
    }

    #[inline]
    pub fn total_bits_per_vec(&self) -> usize {
        *self.bit_offsets.last().unwrap()
    }

    /// Encode a batch of vectors (`n * dim` floats) into bit-packed codes.
    pub fn encode(&self, vectors: &[f32]) -> Vec<u8> {
        let d = self.cfg.dim;
        assert_eq!(vectors.len() % d, 0);
        let n = vectors.len() / d;
        let mut out = vec![0u8; n * self.bytes_per_vec];
        for i in 0..n {
            self.encode_one(&vectors[i * d..(i + 1) * d], &mut out[i * self.bytes_per_vec..(i + 1) * self.bytes_per_vec]);
        }
        out
    }

    fn encode_one(&self, v: &[f32], out: &mut [u8]) {
        let d = self.cfg.dim;
        for byte in out.iter_mut() {
            *byte = 0;
        }
        for j in 0..d {
            let bits = self.bits[j];
            if bits == 0 {
                continue;
            }
            let levels_max = (1u32 << bits) - 1;
            let x = v[j];
            let step = self.steps[j];
            let min = self.mins[j];
            let q = (((x - min) / step).round()).max(0.0) as u32;
            let q = q.min(levels_max);
            write_bits(out, self.bit_offsets[j], bits as usize, q as u64);
        }
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let d = self.cfg.dim;
        let mut out = vec![0.0_f32; d];
        for j in 0..d {
            let bits = self.bits[j] as usize;
            if bits == 0 {
                out[j] = self.mins[j];
                continue;
            }
            let q = read_bits(code, self.bit_offsets[j], bits);
            out[j] = self.mins[j] + (q as f32) * self.steps[j];
        }
        out
    }

    /// Return a per-dim LUT of reconstructed scalar values indexed by the
    /// quantized code. Used by `distance_lut`.
    pub fn build_lut(&self) -> Vec<Vec<f32>> {
        let d = self.cfg.dim;
        let mut lut = Vec::with_capacity(d);
        for j in 0..d {
            let levels = 1usize << self.bits[j] as usize;
            let mut row = Vec::with_capacity(levels);
            for q in 0..levels {
                row.push(self.mins[j] + (q as f32) * self.steps[j]);
            }
            lut.push(row);
        }
        lut
    }

    /// Compute squared L2 distance between a float query and one encoded
    /// vector using the per-dim LUT. The query-side cost is a (b_j)-bit
    /// unpack per dim + table lookup + fma, which is the hot loop that
    /// decides whether VA-SQ pays off versus uniform SQ.
    #[inline]
    pub fn distance_lut(&self, query: &[f32], code: &[u8], lut: &[Vec<f32>]) -> f32 {
        let d = self.cfg.dim;
        let mut acc = 0.0_f32;
        for j in 0..d {
            let bits = self.bits[j] as usize;
            let q = if bits == 0 {
                0
            } else {
                read_bits(code, self.bit_offsets[j], bits) as usize
            };
            let rec = lut[j][q];
            let diff = query[j] - rec;
            acc += diff * diff;
        }
        acc
    }

    /// Bytes used per vector by the codec itself (not counting the LUT,
    /// which is per-codec, not per-vector).
    pub fn bytes_per_vec(&self) -> usize {
        self.bytes_per_vec
    }

    pub fn average_bits_per_dim(&self) -> f32 {
        self.total_bits_per_vec() as f32 / self.cfg.dim as f32
    }
}

#[inline]
fn write_bits(buf: &mut [u8], offset: usize, nbits: usize, value: u64) {
    let mut remaining = nbits;
    let mut off = offset;
    let mut v = value;
    while remaining > 0 {
        let byte = off / 8;
        let bit = off % 8;
        let room = 8 - bit;
        let take = remaining.min(room);
        let mask = ((1u64 << take) - 1) as u8;
        let chunk = (v as u8) & mask;
        buf[byte] |= chunk << bit;
        v >>= take;
        off += take;
        remaining -= take;
    }
}

#[inline]
fn read_bits(buf: &[u8], offset: usize, nbits: usize) -> u64 {
    let mut remaining = nbits;
    let mut off = offset;
    let mut out = 0u64;
    let mut shift = 0u32;
    while remaining > 0 {
        let byte = off / 8;
        let bit = off % 8;
        let room = 8 - bit;
        let take = remaining.min(room);
        let mask = ((1u64 << take) - 1) as u8;
        let chunk = ((buf[byte] >> bit) & mask) as u64;
        out |= chunk << shift;
        shift += take as u32;
        off += take;
        remaining -= take;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::AllocPolicy;

    #[test]
    fn roundtrip_uniform() {
        let d = 32;
        let n = 128;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                data.push(((i * 31 + j * 7) as f32).sin());
            }
        }
        let codec = VaSqCodec::train(
            &data,
            VaSqConfig {
                dim: d,
                budget_bits_per_dim: 8,
                b_min: 8,
                b_max: 8,
                policy: AllocPolicy::Uniform,
            },
        );
        let codes = codec.encode(&data);
        let v0 = codec.decode(&codes[0..codec.bytes_per_vec()]);
        for j in 0..d {
            assert!((v0[j] - data[j]).abs() < codec.steps[j] * 1.01);
        }
    }

    #[test]
    fn log_variance_puts_more_bits_on_hi_variance_dims() {
        let d = 16;
        let n = 2000;
        let mut data = Vec::with_capacity(n * d);
        // Half the dims have variance 1, the other half have variance 100.
        let mut state: u32 = 42;
        for _ in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (state >> 8) as f32 / ((1u32 << 24) as f32);
                let z = (u - 0.5) * 2.0;
                let scale = if j < d / 2 { 1.0 } else { 100.0 };
                data.push(z * scale);
            }
        }
        let codec = VaSqCodec::train(
            &data,
            VaSqConfig {
                dim: d,
                budget_bits_per_dim: 4,
                b_min: 1,
                b_max: 8,
                policy: AllocPolicy::LogVariance,
            },
        );
        let hi: u32 = codec.bits[d / 2..].iter().sum();
        let lo: u32 = codec.bits[..d / 2].iter().sum();
        assert!(hi >= lo, "high-variance half {} should get at least as many bits as low {}", hi, lo);
    }

    #[test]
    fn lut_distance_matches_decoded_l2() {
        let d = 24;
        let n = 64;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                data.push(((i * 13 + j * 5) as f32).cos());
            }
        }
        let codec = VaSqCodec::train(
            &data,
            VaSqConfig {
                dim: d,
                budget_bits_per_dim: 6,
                b_min: 2,
                b_max: 8,
                policy: AllocPolicy::LogVariance,
            },
        );
        let codes = codec.encode(&data);
        let lut = codec.build_lut();
        let query = data[0..d].to_vec();
        for i in 0..n {
            let code = &codes[i * codec.bytes_per_vec()..(i + 1) * codec.bytes_per_vec()];
            let rec = codec.decode(code);
            let direct: f32 = (0..d).map(|j| (query[j] - rec[j]).powi(2)).sum();
            let via_lut = codec.distance_lut(&query, code, &lut);
            assert!((direct - via_lut).abs() < 1e-4, "mismatch at {}: direct={} lut={}", i, direct, via_lut);
        }
    }
}
