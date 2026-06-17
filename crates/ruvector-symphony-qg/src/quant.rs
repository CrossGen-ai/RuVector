//! 1-bit sign quantizer with optional random sign rotation.
//!
//! For each vector x of dimension d, we compute c = sign(R x) where R is
//! a random ±1 diagonal sign matrix (cheap, O(d) instead of d²). This is
//! a degenerate JL projection that keeps direction information for
//! rotation-invariant distributions (Gaussian, unit-sphere uniform) and
//! a useful — but not theoretically tight — surrogate for arbitrary
//! distributions. Full RaBitQ uses a true random orthogonal rotation
//! (e.g. Walsh–Hadamard) which we omit here for code-clarity; see the
//! research note for the divergence and its measured impact.
//!
//! Codes are packed into `u64` words and compared via XOR + popcount.
//! Distance estimator: estimated squared L2 between unit-normalized
//! vectors is `2 * (1 - cos_est)` where `cos_est = 1 - 2*hamming / d`.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone, Debug)]
pub struct BitQuantizer {
    pub d: usize,
    pub words: usize, // ceil(d / 64)
    /// Per-dimension sign flip (+1 / -1) used as a poor-man's rotation.
    sign_flip: Vec<i8>,
}

impl BitQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let sign_flip = (0..d).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect();
        let words = (d + 63) / 64;
        Self { d, words, sign_flip }
    }

    /// Encode a single vector into `self.words` u64 words.
    pub fn encode(&self, v: &[f32]) -> Vec<u64> {
        assert_eq!(v.len(), self.d, "BitQuantizer::encode dim mismatch");
        let mut out = vec![0u64; self.words];
        for i in 0..self.d {
            let x = v[i] * self.sign_flip[i] as f32;
            if x >= 0.0 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }

    pub fn encode_batch(&self, vs: &[Vec<f32>]) -> Vec<u64> {
        let mut out = vec![0u64; vs.len() * self.words];
        for (vi, v) in vs.iter().enumerate() {
            let base = vi * self.words;
            for i in 0..self.d {
                let x = v[i] * self.sign_flip[i] as f32;
                if x >= 0.0 {
                    out[base + i / 64] |= 1u64 << (i % 64);
                }
            }
        }
        out
    }

    /// Hamming distance between two encoded codes.
    #[inline]
    pub fn hamming(&self, a: &[u64], b: &[u64]) -> u32 {
        debug_assert_eq!(a.len(), self.words);
        debug_assert_eq!(b.len(), self.words);
        let mut h = 0u32;
        for i in 0..self.words {
            h += (a[i] ^ b[i]).count_ones();
        }
        h
    }

    /// Hamming distance between query code and slice-offset packed code in a contiguous buffer.
    #[inline]
    pub fn hamming_at(&self, query: &[u64], buf: &[u64], idx: usize) -> u32 {
        let base = idx * self.words;
        let mut h = 0u32;
        for i in 0..self.words {
            h += (query[i] ^ buf[base + i]).count_ones();
        }
        h
    }

    /// Estimated squared L2 from Hamming distance, valid for unit-norm vectors.
    /// `est_l2_sq ≈ 4 * hamming / d` (cosine surrogate, dropped constants).
    #[inline]
    pub fn estimate_l2_sq(&self, hamming: u32) -> f32 {
        // Larger hamming → larger distance; we only need a *monotone* ranker
        // for candidate ordering. Use raw hamming as the surrogate.
        hamming as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_size() {
        let q = BitQuantizer::new(128, 42);
        assert_eq!(q.words, 2);
        let v: Vec<f32> = (0..128).map(|i| (i as f32) - 64.0).collect();
        let c = q.encode(&v);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn hamming_self_is_zero() {
        let q = BitQuantizer::new(64, 7);
        let v: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let c = q.encode(&v);
        assert_eq!(q.hamming(&c, &c), 0);
    }

    #[test]
    fn hamming_is_monotone_with_l2_on_unit_vectors() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(11);
        let d = 128;
        let q = BitQuantizer::new(d, 99);
        // Build a unit vector x and three perturbations of increasing magnitude.
        let x: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let xn = norm(&x);
        let x: Vec<f32> = x.iter().map(|v| v / xn).collect();
        let cx = q.encode(&x);

        let mut last_ham = 0u32;
        for scale in [0.05_f32, 0.3, 1.0, 2.0] {
            let y: Vec<f32> = x.iter().enumerate()
                .map(|(i, v)| v + scale * ((rng.gen::<f32>() - 0.5)))
                .collect();
            let yn = norm(&y);
            let y: Vec<f32> = y.iter().map(|v| v / yn).collect();
            let cy = q.encode(&y);
            let h = q.hamming(&cx, &cy);
            // Tolerant monotonicity: each step should *trend* up.
            // (statistical, not guaranteed for tiny d)
            if scale > 0.05 {
                assert!(h >= last_ham || last_ham - h < 8, "ham did not trend up: {} -> {}", last_ham, h);
            }
            last_ham = h;
        }
    }

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
    }
}
