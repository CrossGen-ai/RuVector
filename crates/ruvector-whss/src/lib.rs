//! Walsh-Hadamard Sign Sketch (WHSS) prefilter for approximate nearest neighbor search.
//!
//! Pipeline per vector x (dim d, padded to pow2 D):
//!   1. componentwise random sign flip: y = diag(s) * x
//!   2. in-place Fast Walsh-Hadamard Transform: y' = H_D * y
//!   3. take sign bits of the first `m` coefficients -> m-bit sketch
//!
//! Search:
//!   * compute query sketch once (O(D log D))
//!   * rank corpus by Hamming distance via SIMD popcount (O(N * m/64))
//!   * verify top-c candidates with full L2 (O(c * d))

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Round up to the next power of two (minimum 1).
#[inline]
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// In-place Fast Walsh-Hadamard Transform (unnormalized Hadamard-ordering).
/// `v.len()` must be a power of two.
pub fn fwht(v: &mut [f32]) {
    let n = v.len();
    assert!(n.is_power_of_two(), "FWHT length must be power of two");
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        let mut i = 0usize;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
}

/// A trained WHSS projector.
#[derive(Clone)]
pub struct Whss {
    /// Padded transform dim (pow2, >= input dim).
    pub dim_pad: usize,
    /// Input dim.
    pub dim_in: usize,
    /// Number of sketch bits (<= dim_pad, multiple of 64 recommended).
    pub m_bits: usize,
    /// Random sign diagonal: +1.0 / -1.0 per padded coordinate.
    pub signs: Vec<f32>,
    /// A permutation of [0, dim_pad) picking which m coefficients become sketch bits.
    pub pick: Vec<u32>,
}

impl Whss {
    pub fn new(dim_in: usize, m_bits: usize, seed: u64) -> Self {
        assert!(m_bits % 64 == 0 && m_bits > 0, "m_bits must be a positive multiple of 64");
        let dim_pad = next_pow2(dim_in.max(m_bits));
        assert!(m_bits <= dim_pad);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..dim_pad).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        // Fisher-Yates on 0..dim_pad, take first m_bits.
        let mut perm: Vec<u32> = (0..dim_pad as u32).collect();
        for i in (1..perm.len()).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        perm.truncate(m_bits);
        Self { dim_pad, dim_in, m_bits, signs, pick: perm }
    }

    /// Project one vector into an m-bit sketch stored as u64 words.
    pub fn sketch(&self, x: &[f32], out: &mut [u64]) {
        assert_eq!(x.len(), self.dim_in);
        assert_eq!(out.len(), self.m_bits / 64);
        let mut buf = vec![0f32; self.dim_pad];
        for (i, &xi) in x.iter().enumerate() {
            buf[i] = xi * self.signs[i];
        }
        // apply remaining signs to padding (zeroes stay zero but that's fine)
        for i in self.dim_in..self.dim_pad {
            buf[i] = 0.0;
        }
        fwht(&mut buf);
        for w in out.iter_mut() {
            *w = 0;
        }
        for (b, &pi) in self.pick.iter().enumerate() {
            if buf[pi as usize] >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
    }

    pub fn words_per_sketch(&self) -> usize {
        self.m_bits / 64
    }
}

/// Encode a corpus of vectors to a flat Vec<u64> of length N * words_per_sketch.
pub fn encode_corpus(whss: &Whss, corpus: &[Vec<f32>]) -> Vec<u64> {
    let w = whss.words_per_sketch();
    let mut out = vec![0u64; corpus.len() * w];
    for (i, v) in corpus.iter().enumerate() {
        whss.sketch(v, &mut out[i * w..(i + 1) * w]);
    }
    out
}

/// Hamming distance between two sketches (same word count).
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    let mut s = 0u32;
    for i in 0..a.len() {
        s += (a[i] ^ b[i]).count_ones();
    }
    s
}

/// Squared L2 distance.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Full-scan exact L2 top-k. Returns indices sorted ascending by distance.
pub fn exact_topk(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = corpus.iter().enumerate().map(|(i, v)| (l2sq(v, q), i)).collect();
    scored.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.truncate(k);
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().map(|(_, i)| i).collect()
}

/// WHSS prefilter + exact verify. Returns top-k by true L2 among the Hamming top-c.
pub fn whss_search(
    whss: &Whss,
    corpus: &[Vec<f32>],
    sketches: &[u64],
    q: &[f32],
    k: usize,
    cand: usize,
) -> Vec<usize> {
    let w = whss.words_per_sketch();
    let mut qsk = vec![0u64; w];
    whss.sketch(q, &mut qsk);
    let n = corpus.len();
    // rank by Hamming
    let mut cand_scored: Vec<(u32, usize)> = (0..n).map(|i| (hamming(&qsk, &sketches[i * w..(i + 1) * w]), i)).collect();
    let c = cand.min(n);
    cand_scored.select_nth_unstable_by_key(c - 1, |x| x.0);
    cand_scored.truncate(c);
    // verify
    let mut ver: Vec<(f32, usize)> = cand_scored.into_iter().map(|(_, i)| (l2sq(&corpus[i], q), i)).collect();
    let kk = k.min(ver.len());
    ver.select_nth_unstable_by(kk - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    ver.truncate(kk);
    ver.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ver.into_iter().map(|(_, i)| i).collect()
}

// ---- Baseline: Gaussian sign-random-projection ----

/// Classic SRP: y = sign(G * x), G ~ N(0, 1), m rows.
#[derive(Clone)]
pub struct Srp {
    pub dim_in: usize,
    pub m_bits: usize,
    pub g: Vec<f32>, // row-major m x dim_in
}

impl Srp {
    pub fn new(dim_in: usize, m_bits: usize, seed: u64) -> Self {
        assert!(m_bits % 64 == 0);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut g = Vec::with_capacity(m_bits * dim_in);
        for _ in 0..(m_bits * dim_in) {
            // Box-Muller
            let u1: f32 = rng.gen_range(1e-9f32..1.0);
            let u2: f32 = rng.gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            g.push(z);
        }
        Self { dim_in, m_bits, g }
    }

    pub fn words_per_sketch(&self) -> usize {
        self.m_bits / 64
    }

    pub fn sketch(&self, x: &[f32], out: &mut [u64]) {
        assert_eq!(x.len(), self.dim_in);
        assert_eq!(out.len(), self.m_bits / 64);
        for w in out.iter_mut() {
            *w = 0;
        }
        for b in 0..self.m_bits {
            let row = &self.g[b * self.dim_in..(b + 1) * self.dim_in];
            let mut dot = 0f32;
            for i in 0..self.dim_in {
                dot += row[i] * x[i];
            }
            if dot >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
    }
}

pub fn srp_encode_corpus(srp: &Srp, corpus: &[Vec<f32>]) -> Vec<u64> {
    let w = srp.words_per_sketch();
    let mut out = vec![0u64; corpus.len() * w];
    for (i, v) in corpus.iter().enumerate() {
        srp.sketch(v, &mut out[i * w..(i + 1) * w]);
    }
    out
}

pub fn srp_search(
    srp: &Srp,
    corpus: &[Vec<f32>],
    sketches: &[u64],
    q: &[f32],
    k: usize,
    cand: usize,
) -> Vec<usize> {
    let w = srp.words_per_sketch();
    let mut qsk = vec![0u64; w];
    srp.sketch(q, &mut qsk);
    let n = corpus.len();
    let mut cand_scored: Vec<(u32, usize)> = (0..n).map(|i| (hamming(&qsk, &sketches[i * w..(i + 1) * w]), i)).collect();
    let c = cand.min(n);
    cand_scored.select_nth_unstable_by_key(c - 1, |x| x.0);
    cand_scored.truncate(c);
    let mut ver: Vec<(f32, usize)> = cand_scored.into_iter().map(|(_, i)| (l2sq(&corpus[i], q), i)).collect();
    let kk = k.min(ver.len());
    ver.select_nth_unstable_by(kk - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    ver.truncate(kk);
    ver.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ver.into_iter().map(|(_, i)| i).collect()
}

/// Recall@k given ground-truth top-k and predicted top-k.
pub fn recall_at_k(truth: &[usize], pred: &[usize]) -> f32 {
    let k = truth.len();
    let truth_set: std::collections::HashSet<usize> = truth.iter().copied().collect();
    let hits = pred.iter().filter(|p| truth_set.contains(p)).count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_self_inverse() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = v.clone();
        fwht(&mut v);
        fwht(&mut v);
        // H*H = N*I
        for i in 0..v.len() {
            v[i] /= v.len() as f32;
        }
        for i in 0..orig.len() {
            assert!((v[i] - orig[i]).abs() < 1e-4, "mismatch at {}: {} vs {}", i, v[i], orig[i]);
        }
    }

    #[test]
    fn sketch_is_deterministic() {
        let whss = Whss::new(64, 128, 42);
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.013 - 0.5).collect();
        let mut a = vec![0u64; whss.words_per_sketch()];
        let mut b = vec![0u64; whss.words_per_sketch()];
        whss.sketch(&x, &mut a);
        whss.sketch(&x, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn hamming_tracks_similarity() {
        // Nearly-identical vectors should have small Hamming; orthogonal-ish should have ~m/2.
        let whss = Whss::new(128, 256, 7);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mut v2 = v.clone();
        for i in 0..128 {
            v2[i] += (rng.gen::<f32>() - 0.5) * 0.01;
        }
        let v3: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
        let w = whss.words_per_sketch();
        let mut s1 = vec![0u64; w];
        let mut s2 = vec![0u64; w];
        let mut s3 = vec![0u64; w];
        whss.sketch(&v, &mut s1);
        whss.sketch(&v2, &mut s2);
        whss.sketch(&v3, &mut s3);
        let close = hamming(&s1, &s2);
        let far = hamming(&s1, &s3);
        assert!(close < far, "close={} far={}", close, far);
        assert!(close < 32, "near-identical vectors produced Hamming {}", close);
    }

    #[test]
    fn recall_positive_on_synthetic() {
        let mut rng = ChaCha20Rng::seed_from_u64(1234);
        let d = 128;
        let n = 500;
        let corpus: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        let whss = Whss::new(d, 256, 11);
        let sk = encode_corpus(&whss, &corpus);
        let mut total_recall = 0.0;
        let k = 10;
        let qcount = 20;
        for _ in 0..qcount {
            let q: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
            let truth = exact_topk(&corpus, &q, k);
            let pred = whss_search(&whss, &corpus, &sk, &q, k, 100);
            total_recall += recall_at_k(&truth, &pred);
        }
        let avg = total_recall / qcount as f32;
        assert!(avg >= 0.5, "WHSS recall@10 too low on synthetic: {}", avg);
    }
}
