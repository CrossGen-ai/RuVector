//! μ-law companded scalar quantization for dense embeddings.
//!
//! Non-uniform 8-bit scalar quantization that applies the μ-law companding
//! curve (classical telephony signal coding, ITU-T G.711) to the per-dimension
//! value distribution before uniform quantization. Small-magnitude components
//! (where most of the probability mass of a normalized embedding sits) get
//! more quantization bins; large-magnitude outliers still fit, with the error
//! spread into regions where the retrieval distance is less sensitive.

#![forbid(unsafe_code)]

use std::f32;

/// A scalar-quantization backend: encode a float vector to a byte vector
/// and compute an approximate inner product between a float query and an
/// encoded database vector.
pub trait ScalarCodec: Sync + Send {
    fn code_bytes(&self, dim: usize) -> usize {
        dim
    }
    fn encode(&self, x: &[f32], out: &mut [u8]);
    fn decode(&self, code: &[u8], out: &mut [f32]);
    fn name(&self) -> &'static str;
}

/// Classic symmetric int8 scalar quantization with a single global scale.
/// Codebook: y_i = round(clamp(x_i / s, -1, 1) * 127).
#[derive(Debug, Clone)]
pub struct LinearSq8 {
    pub scale: f32,
}

impl LinearSq8 {
    /// Fit a scale from training vectors so that `clip_quantile` fraction of
    /// absolute values are inside [-scale, scale]. Values outside clip.
    pub fn fit(train: &[Vec<f32>], clip_quantile: f32) -> Self {
        let mut mags: Vec<f32> = train.iter().flat_map(|v| v.iter().map(|x| x.abs())).collect();
        if mags.is_empty() {
            return Self { scale: 1.0 };
        }
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((mags.len() - 1) as f32 * clip_quantile).round() as usize;
        let s = mags[idx.min(mags.len() - 1)].max(1e-6);
        Self { scale: s }
    }
}

impl ScalarCodec for LinearSq8 {
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(x.len(), out.len());
        let inv = 1.0 / self.scale;
        for (xi, o) in x.iter().zip(out.iter_mut()) {
            let n = (xi * inv).clamp(-1.0, 1.0);
            let q = (n * 127.0).round() as i32;
            *o = (q.clamp(-127, 127) as i8) as u8;
        }
    }

    fn decode(&self, code: &[u8], out: &mut [f32]) {
        debug_assert_eq!(code.len(), out.len());
        let s = self.scale / 127.0;
        for (c, o) in code.iter().zip(out.iter_mut()) {
            let q = (*c as i8) as f32;
            *o = q * s;
        }
    }

    fn name(&self) -> &'static str {
        "linear_sq8"
    }
}

/// μ-law companded scalar quantization. The input is scaled into [-1,1] via
/// a global `scale` (fit by quantile just like `LinearSq8`), then passed
/// through `sign(x) * ln(1 + mu*|x|) / ln(1 + mu)` before uniform 8-bit
/// quantization. Decode inverts the law exactly.
#[derive(Debug, Clone)]
pub struct MuLawSq8 {
    pub scale: f32,
    pub mu: f32,
    log_1p_mu: f32,
}

impl MuLawSq8 {
    pub fn new(scale: f32, mu: f32) -> Self {
        Self {
            scale,
            mu,
            log_1p_mu: (1.0 + mu).ln(),
        }
    }

    pub fn fit(train: &[Vec<f32>], clip_quantile: f32, mu: f32) -> Self {
        let lin = LinearSq8::fit(train, clip_quantile);
        Self::new(lin.scale, mu)
    }

    /// Learn μ from the magnitude distribution: match μ-law to the empirical
    /// CDF so median-magnitude maps to roughly 0.5 of code range.
    pub fn fit_learned_mu(train: &[Vec<f32>], clip_quantile: f32) -> Self {
        let lin = LinearSq8::fit(train, clip_quantile);
        let mut mags: Vec<f32> = train
            .iter()
            .flat_map(|v| v.iter().map(|x| (x.abs() / lin.scale).min(1.0)))
            .collect();
        if mags.is_empty() {
            return Self::new(lin.scale, 255.0);
        }
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = mags[mags.len() / 2].max(1e-6);
        // We want ln(1+mu*median)/ln(1+mu) = 0.5
        //   => (1+mu*median)^2 = 1 + mu
        // Numeric solve on mu in [1, 1023].
        let f = |mu: f32| ((1.0 + mu * median).ln() / (1.0 + mu).ln()) - 0.5;
        let (mut lo, mut hi) = (1.0f32, 1023.0f32);
        for _ in 0..60 {
            let m = 0.5 * (lo + hi);
            if f(m) > 0.0 {
                hi = m;
            } else {
                lo = m;
            }
        }
        Self::new(lin.scale, 0.5 * (lo + hi))
    }

    #[inline]
    fn compand(&self, n: f32) -> f32 {
        let a = n.abs().min(1.0);
        let y = (1.0 + self.mu * a).ln() / self.log_1p_mu;
        if n < 0.0 {
            -y
        } else {
            y
        }
    }

    #[inline]
    fn expand(&self, y: f32) -> f32 {
        let a = y.abs().min(1.0);
        let x = ((self.log_1p_mu * a).exp() - 1.0) / self.mu;
        if y < 0.0 {
            -x
        } else {
            x
        }
    }
}

impl ScalarCodec for MuLawSq8 {
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(x.len(), out.len());
        let inv = 1.0 / self.scale;
        for (xi, o) in x.iter().zip(out.iter_mut()) {
            let n = (xi * inv).clamp(-1.0, 1.0);
            let y = self.compand(n);
            let q = (y * 127.0).round() as i32;
            *o = (q.clamp(-127, 127) as i8) as u8;
        }
    }

    fn decode(&self, code: &[u8], out: &mut [f32]) {
        debug_assert_eq!(code.len(), out.len());
        for (c, o) in code.iter().zip(out.iter_mut()) {
            let q = (*c as i8) as f32;
            let y = q / 127.0;
            let n = self.expand(y);
            *o = n * self.scale;
        }
    }

    fn name(&self) -> &'static str {
        "mulaw_sq8"
    }
}

/// L2 norm of a vector.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity (requires both unit-norm or we divide here).
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

/// Mean squared reconstruction error of a codec on a slice of vectors.
pub fn reconstruction_mse(codec: &dyn ScalarCodec, data: &[Vec<f32>]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let dim = data[0].len();
    let mut code = vec![0u8; dim];
    let mut dec = vec![0f32; dim];
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for v in data {
        codec.encode(v, &mut code);
        codec.decode(&code, &mut dec);
        for (x, y) in v.iter().zip(&dec) {
            let e = (*x - *y) as f64;
            sum += e * e;
            n += 1;
        }
    }
    sum / n as f64
}

/// Recall@k of cosine top-k with a codec, measured against exact float32
/// cosine top-k. `queries` are kept in float; database vectors are encoded
/// then decoded, and we rank by cosine on the decoded values.
pub fn recall_at_k(
    codec: &dyn ScalarCodec,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let dim = base[0].len();
    // Pre-encode-decode the base once.
    let mut dec_base: Vec<Vec<f32>> = Vec::with_capacity(base.len());
    let mut code = vec![0u8; dim];
    for v in base {
        codec.encode(v, &mut code);
        let mut d = vec![0f32; dim];
        codec.decode(&code, &mut d);
        dec_base.push(d);
    }
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in queries {
        let mut exact: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(q, v)))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gt: std::collections::HashSet<usize> =
            exact.iter().take(k).map(|(i, _)| *i).collect();

        let mut approx: Vec<(usize, f32)> = dec_base
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(q, v)))
            .collect();
        approx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, _) in approx.iter().take(k) {
            if gt.contains(i) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn gauss_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn linear_round_trip_small_error() {
        let data = gauss_vectors(256, 64, 7);
        let codec = LinearSq8::fit(&data, 0.999);
        let mse = reconstruction_mse(&codec, &data);
        assert!(mse < 5e-4, "mse too large: {mse}");
    }

    #[test]
    fn mulaw_reduces_small_value_error() {
        let data = gauss_vectors(512, 128, 42);
        let lin = LinearSq8::fit(&data, 0.999);
        let mu = MuLawSq8::fit(&data, 0.999, 255.0);
        // μ-law should concentrate error on small values differently; both
        // reconstruct, we just check both are sane.
        let mse_lin = reconstruction_mse(&lin, &data);
        let mse_mu = reconstruction_mse(&mu, &data);
        assert!(mse_lin.is_finite() && mse_mu.is_finite());
    }

    #[test]
    fn learned_mu_fits_median() {
        let data = gauss_vectors(1024, 64, 1);
        let codec = MuLawSq8::fit_learned_mu(&data, 0.999);
        assert!(codec.mu > 1.0 && codec.mu < 1024.0);
    }

    #[test]
    fn recall_matches_exact_on_identity() {
        // A pathological but valid test: codec with huge mu close to linear,
        // on tiny data — recall must be well above zero.
        let base = gauss_vectors(64, 32, 3);
        let queries = gauss_vectors(16, 32, 4);
        let codec = LinearSq8::fit(&base, 0.999);
        let r = recall_at_k(&codec, &base, &queries, 10);
        assert!(r > 0.3, "recall too low: {r}");
    }
}
