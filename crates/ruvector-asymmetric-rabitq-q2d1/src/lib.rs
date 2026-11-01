//! Asymmetric RaBitQ (q=2 bits, d=1 bit): a query-side ternary code paired
//! with a 1-bit database code, both expressed in a shared Hadamard-rotated
//! basis.
//!
//! Storage per database vector is identical to RaBitQ (1 bit/dim). The query
//! encodes each coordinate into one of {-1, 0, +1} (ternary, ~1.58 bits/dim)
//! so coordinates whose magnitude falls below a per-query threshold are
//! masked out of the inner product instead of being forced to the "wrong"
//! sign. The dot product is a POPCNT of (q.sign XOR d.sign) AND q.mask.
//!
//! Three operating points are exposed so the paper's numbers can be
//! reproduced:
//!
//! * [`Backend::F32Flat`]   — baseline float inner product.
//! * [`Backend::SymRabitQ`] — symmetric 1-bit/1-bit RaBitQ-style estimator.
//! * [`Backend::AsymQ2D1`]  — proposed 2-bit query / 1-bit database.
//!
//! All variants share the same Hadamard rotation and the same ADC-free
//! distance estimator: `||q||·||d||·(1 - 2·H/active)`, where `H` is the
//! Hamming weight of mismatches among `active` query dimensions.

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;

pub const WORDS_PER_VEC_128: usize = 128 / 64;

/// Which estimator a query uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    F32Flat,
    SymRabitQ,
    AsymQ2D1,
}

/// A packed 1-bit code (one bit per dimension), least significant bit = dim 0.
#[derive(Clone, Debug)]
pub struct BitCode {
    pub words: Vec<u64>,
    pub norm: f32,
}

/// A 2-bit ternary query code: `sign` holds the sign bit per dim; `mask`
/// marks "active" dims (those with |x| above the per-query threshold).
#[derive(Clone, Debug)]
pub struct TernaryCode {
    pub sign: Vec<u64>,
    pub mask: Vec<u64>,
    pub norm: f32,
    pub active: u32,
}

/// Rotation applied before encoding. Walsh–Hadamard with ±1 random sign
/// flip — deterministic given the seed, O(d log d), no matrix storage.
#[derive(Clone, Debug)]
pub struct HadamardRotation {
    pub dim: usize,
    pub signs: Vec<f32>,
}

impl HadamardRotation {
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim.is_power_of_two(), "dim must be a power of two");
        let mut rng = SmallRng::seed_from_u64(seed);
        let signs = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        Self { dim, signs }
    }

    /// In-place rotation. `x[i] <- s_i * x[i]` then fast Walsh–Hadamard
    /// (unnormalized). Reciprocal to the inverse up to the `sqrt(dim)` scale
    /// — because we take ratios of inner products the scale cancels.
    pub fn apply(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        for i in 0..self.dim {
            x[i] *= self.signs[i];
        }
        let mut h = 1usize;
        while h < self.dim {
            let step = h * 2;
            let mut i = 0usize;
            while i < self.dim {
                for j in i..i + h {
                    let a = x[j];
                    let b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
                i += step;
            }
            h *= 2;
        }
    }
}

/// Encode a dense f32 vector as a 1-bit sign code.
pub fn encode_bit(x: &[f32]) -> BitCode {
    let dim = x.len();
    let n_words = (dim + 63) / 64;
    let mut words = vec![0u64; n_words];
    let mut sq: f32 = 0.0;
    for (i, &v) in x.iter().enumerate() {
        sq += v * v;
        if v >= 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    BitCode { words, norm: sq.sqrt() }
}

/// Encode a dense f32 vector as a ternary (sign + mask) code. `tau_frac` is
/// the mask threshold as a fraction (0..1) of the median absolute value.
pub fn encode_ternary(x: &[f32], tau_frac: f32) -> TernaryCode {
    let dim = x.len();
    let n_words = (dim + 63) / 64;
    let mut abs: Vec<f32> = x.iter().map(|v| v.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = abs[dim / 2].max(f32::MIN_POSITIVE);
    let tau = tau_frac * median;

    let mut sign = vec![0u64; n_words];
    let mut mask = vec![0u64; n_words];
    let mut sq: f32 = 0.0;
    let mut active = 0u32;
    for (i, &v) in x.iter().enumerate() {
        sq += v * v;
        if v.abs() >= tau {
            mask[i / 64] |= 1u64 << (i % 64);
            active += 1;
            if v >= 0.0 {
                sign[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    TernaryCode { sign, mask, norm: sq.sqrt(), active }
}

/// Flat f32 inner product.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Symmetric 1-bit / 1-bit estimator. Uses full dim as `active`.
#[inline]
pub fn est_sym(q: &BitCode, d: &BitCode, dim: usize) -> f32 {
    let mut h = 0u32;
    for i in 0..q.words.len() {
        h += (q.words[i] ^ d.words[i]).count_ones();
    }
    let agree = dim as f32 - h as f32;
    q.norm * d.norm * (2.0 * agree / dim as f32 - 1.0)
}

/// Asymmetric 2-bit query / 1-bit database estimator.
#[inline]
pub fn est_asym(q: &TernaryCode, d: &BitCode) -> f32 {
    let mut h_active = 0u32;
    for i in 0..q.sign.len() {
        let mismatch = (q.sign[i] ^ d.words[i]) & q.mask[i];
        h_active += mismatch.count_ones();
    }
    if q.active == 0 {
        return 0.0;
    }
    let agree = q.active as f32 - h_active as f32;
    q.norm * d.norm * (2.0 * agree / q.active as f32 - 1.0)
}

/// A tiny in-memory index that stores every variant so a single query can
/// be scored under all three backends on the same data.
pub struct TriIndex {
    pub dim: usize,
    pub rot: HadamardRotation,
    pub flat: Vec<Vec<f32>>,
    pub bit: Vec<BitCode>,
}

impl TriIndex {
    pub fn build(vecs: &[Vec<f32>], seed: u64) -> Self {
        let dim = vecs[0].len();
        let rot = HadamardRotation::new(dim, seed);
        let mut flat = Vec::with_capacity(vecs.len());
        let mut bit = Vec::with_capacity(vecs.len());
        for v in vecs {
            let mut r = v.clone();
            rot.apply(&mut r);
            bit.push(encode_bit(&r));
            flat.push(r);
        }
        Self { dim, rot, flat, bit }
    }

    pub fn topk(&self, query: &[f32], k: usize, backend: Backend, tau_frac: f32) -> Vec<(usize, f32)> {
        let mut q = query.to_vec();
        self.rot.apply(&mut q);

        let (scores, _tag): (Vec<(usize, f32)>, &'static str) = match backend {
            Backend::F32Flat => (
                self.flat
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (i, dot_f32(&q, d)))
                    .collect(),
                "f32",
            ),
            Backend::SymRabitQ => {
                let qc = encode_bit(&q);
                (
                    self.bit
                        .iter()
                        .enumerate()
                        .map(|(i, d)| (i, est_sym(&qc, d, self.dim)))
                        .collect(),
                    "sym",
                )
            }
            Backend::AsymQ2D1 => {
                let qc = encode_ternary(&q, tau_frac);
                (
                    self.bit
                        .iter()
                        .enumerate()
                        .map(|(i, d)| (i, est_asym(&qc, d)))
                        .collect(),
                    "asym",
                )
            }
        };

        let mut s = scores;
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        s.truncate(k);
        s
    }
}

/// Deterministic Gaussian corpus for the demo + tests.
pub fn gen_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
        .collect()
}

/// Recall@k of `pred` against `truth` (set intersection / k).
pub fn recall_at_k(pred: &[(usize, f32)], truth: &[(usize, f32)], k: usize) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<usize> = truth.iter().take(k).map(|p| p.0).collect();
    let hits = pred.iter().take(k).filter(|p| t.contains(&p.0)).count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_is_orthogonal_up_to_scale() {
        let rot = HadamardRotation::new(16, 7);
        let mut x = vec![1.0; 16];
        let orig_norm_sq: f32 = x.iter().map(|v| v * v).sum();
        rot.apply(&mut x);
        let rot_norm_sq: f32 = x.iter().map(|v| v * v).sum();
        // WHT (unnormalized) scales norm^2 by `dim`.
        assert!((rot_norm_sq - orig_norm_sq * 16.0).abs() < 1e-3);
    }

    #[test]
    fn ternary_mask_matches_active_count() {
        let x: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 2.0 } else { 0.0 }).collect();
        let c = encode_ternary(&x, 0.5);
        assert_eq!(c.active, 32);
    }

    #[test]
    fn asym_beats_sym_on_recall_at_10() {
        // Small deterministic corpus: assert the asymmetric estimator
        // retrieves at least as much as the symmetric one on average.
        let corpus = gen_corpus(2_000, 128, 42);
        let idx = TriIndex::build(&corpus, 11);
        let queries = gen_corpus(50, 128, 99);

        let mut sym_total = 0.0f32;
        let mut asym_total = 0.0f32;
        for q in &queries {
            let truth = idx.topk(q, 10, Backend::F32Flat, 0.0);
            let sym = idx.topk(q, 10, Backend::SymRabitQ, 0.0);
            let asym = idx.topk(q, 10, Backend::AsymQ2D1, 0.5);
            sym_total += recall_at_k(&sym, &truth, 10);
            asym_total += recall_at_k(&asym, &truth, 10);
        }
        let n = queries.len() as f32;
        let sym_r = sym_total / n;
        let asym_r = asym_total / n;
        // The asymmetric estimator must not degrade recall.
        assert!(asym_r + 0.005 >= sym_r,
            "asym recall {asym_r:.3} unexpectedly worse than sym {sym_r:.3}");
    }
}
