//! Three swappable NF4 variants.

use crate::codebook::Codebook16;
use crate::quantizer::{Code4, Quantizer};

/// **Variant 1** — Symmetric NF4 with fixed codebook (16 quantiles of
/// N(0,1) normalized to `[-1, 1]`). One global scale for the dataset.
#[derive(Debug, Clone)]
pub struct FixedNf4 {
    pub cb: Codebook16,
    pub global_scale: f32,
}

impl FixedNf4 {
    pub fn new(global_scale: f32) -> Self {
        Self {
            cb: Codebook16::symmetric_normal(),
            global_scale: global_scale.max(1e-12),
        }
    }

    /// Choose `global_scale` as the empirical standard deviation of a
    /// sample, which maps the N(0,1) codebook onto the data.
    pub fn fit_global_scale(sample: &[f32]) -> Self {
        let n = sample.len().max(1) as f64;
        let mean = sample.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = sample
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        // levels live in [-1, 1]; set scale so ±3σ maps to ±1.
        let sigma = var.sqrt().max(1e-12);
        Self::new((3.0 * sigma) as f32)
    }
}

impl Quantizer for FixedNf4 {
    fn encode(&self, x: &[f32]) -> Code4 {
        let mut c = Code4::new(x.len(), self.global_scale, 0.0);
        let inv = 1.0 / self.global_scale;
        for (i, &xi) in x.iter().enumerate() {
            let norm = (xi * inv).clamp(-1.0, 1.0);
            c.set_nibble(i, self.cb.encode(norm));
        }
        c
    }

    fn decode_into(&self, code: &Code4, out: &mut [f32]) {
        debug_assert_eq!(out.len(), code.dim as usize);
        for i in 0..code.dim as usize {
            out[i] = code.scale * self.cb.levels[code.get_nibble(i) as usize];
        }
    }

    #[inline]
    fn level_at(&self, k: u8) -> f32 {
        self.cb.levels[(k & 0x0F) as usize]
    }
}

/// **Variant 2** — QLoRA-style NF4 with per-vector max-abs scale.
/// Mirrors the original NF4 data type for weights, applied to vectors.
#[derive(Debug, Clone)]
pub struct AsymmetricNf4 {
    pub cb: Codebook16,
}

impl Default for AsymmetricNf4 {
    fn default() -> Self {
        Self::new()
    }
}

impl AsymmetricNf4 {
    pub fn new() -> Self {
        Self { cb: Codebook16::qlora_nf4() }
    }
}

impl Quantizer for AsymmetricNf4 {
    fn encode(&self, x: &[f32]) -> Code4 {
        let m = x.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-12);
        let mut c = Code4::new(x.len(), m, 0.0);
        let inv = 1.0 / m;
        for (i, &xi) in x.iter().enumerate() {
            let norm = xi * inv;
            c.set_nibble(i, self.cb.encode(norm));
        }
        c
    }

    fn decode_into(&self, code: &Code4, out: &mut [f32]) {
        debug_assert_eq!(out.len(), code.dim as usize);
        for i in 0..code.dim as usize {
            out[i] = code.scale * self.cb.levels[code.get_nibble(i) as usize];
        }
    }

    #[inline]
    fn level_at(&self, k: u8) -> f32 {
        self.cb.levels[(k & 0x0F) as usize]
    }
}

/// **Variant 3** — Learned-quantile NF4 with per-vector scale+bias.
/// Codebook is fit with Lloyd-Max on a representative sample. Relaxes
/// the "exactly normal" assumption. Encoded with asymmetric
/// min/max → scale+bias to center the data on `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct LearnedNf4 {
    pub cb: Codebook16,
}

impl LearnedNf4 {
    /// Fit the codebook on a representative sample (all values
    /// concatenated). The sample should already have been
    /// normalized per-vector by min/max if the caller wants to match
    /// encode behavior; this is what the benchmark driver does.
    pub fn fit(sample: &[f32], iters: usize) -> Self {
        Self { cb: Codebook16::fit_lloyd_max(sample, iters) }
    }
}

impl Quantizer for LearnedNf4 {
    fn encode(&self, x: &[f32]) -> Code4 {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in x {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        // Map [mn, mx] → [-1, 1].
        let span = (mx - mn).max(1e-12);
        let scale = span * 0.5;
        let bias = (mn + mx) * 0.5;
        let mut c = Code4::new(x.len(), scale, bias);
        let inv = 1.0 / scale;
        for (i, &xi) in x.iter().enumerate() {
            let norm = ((xi - bias) * inv).clamp(-1.0, 1.0);
            c.set_nibble(i, self.cb.encode(norm));
        }
        c
    }

    fn decode_into(&self, code: &Code4, out: &mut [f32]) {
        debug_assert_eq!(out.len(), code.dim as usize);
        for i in 0..code.dim as usize {
            out[i] = code.scale * self.cb.levels[code.get_nibble(i) as usize]
                + code.bias;
        }
    }

    #[inline]
    fn level_at(&self, k: u8) -> f32 {
        self.cb.levels[(k & 0x0F) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim)
            .map(|_| {
                let s: f32 = StandardNormal.sample(&mut rng);
                s
            })
            .collect()
    }

    #[test]
    fn fixed_roundtrip_bounded_error() {
        let v = random_vec(256, 1);
        let q = FixedNf4::fit_global_scale(&v);
        let code = q.encode(&v);
        let mut out = vec![0f32; v.len()];
        q.decode_into(&code, &mut out);
        let err: f32 = v
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / v.len() as f32;
        assert!(err < 0.05, "fixed-NF4 MSE should be small, got {}", err);
    }

    #[test]
    fn asym_roundtrip_bounded_error() {
        let v = random_vec(256, 2);
        let q = AsymmetricNf4::new();
        let code = q.encode(&v);
        let mut out = vec![0f32; v.len()];
        q.decode_into(&code, &mut out);
        let err: f32 = v
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / v.len() as f32;
        assert!(err < 0.05, "asym-NF4 MSE should be small, got {}", err);
    }

    #[test]
    fn learned_roundtrip_bounded_error() {
        // Flat/uniform distribution: learned should beat fixed-normal here.
        let mut rng = StdRng::seed_from_u64(3);
        let v: Vec<f32> =
            (0..256).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let q = LearnedNf4::fit(&v, 20);
        let code = q.encode(&v);
        let mut out = vec![0f32; v.len()];
        q.decode_into(&code, &mut out);
        let err: f32 = v
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / v.len() as f32;
        assert!(err < 0.01, "learned-NF4 MSE on uniform should be small, got {}", err);
    }

    #[test]
    fn asym_l2_matches_decoded_distance() {
        let v = random_vec(128, 4);
        let qvec = random_vec(128, 5);
        let q = AsymmetricNf4::new();
        let code = q.encode(&v);
        let mut rec = vec![0f32; v.len()];
        q.decode_into(&code, &mut rec);
        let direct: f32 = qvec
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let via_trait = q.asym_l2_sq(&qvec, &code);
        assert!((direct - via_trait).abs() < 1e-3, "{} vs {}", direct, via_trait);
    }
}
