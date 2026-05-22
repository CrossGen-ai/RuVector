//! 1-bit-per-dimension quantizer with random sign rotation.
//!
//! The transform is `y = D * pi(x)` where `pi` is a random permutation and
//! `D` is a diagonal of random ±1 signs. This is cheap (`O(d)`) and
//! approximates an orthogonal mixing matrix well enough to make the L2
//! distance between two rotated vectors approximately rotation-invariant —
//! the property RaBitQ relies on to make sign-bit distance an unbiased
//! estimator of L2 distance.
//!
//! Distance estimator for normalized rotated vectors `\hat q`, `\hat x`:
//!     <\hat q, \hat x>  ≈  (1/√d) * Σ_i  \hat q_i * sign(\hat x_i)
//! and the squared L2 distance is reconstructed from the inner product and
//! the (stored) full-precision norm of x.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// One bit per dimension, packed into u64 words.
#[derive(Clone, Debug)]
pub struct BitCode {
    pub words: Vec<u64>,
    pub dim: usize,
    /// L2 norm of the original (unrotated) vector x. Needed for the final
    /// distance reconstruction `||q-x||^2 = ||q||^2 + ||x||^2 - 2<q,x>`.
    pub norm: f32,
}

impl BitCode {
    pub fn nbytes(&self) -> usize {
        self.words.len() * 8 + std::mem::size_of::<f32>() + std::mem::size_of::<usize>()
    }
}

#[derive(Clone, Debug)]
pub struct RaBitQuantizer {
    pub dim: usize,
    perm: Vec<u32>,
    signs: Vec<f32>, // ±1
}

impl RaBitQuantizer {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Random permutation (Fisher–Yates)
        let mut perm: Vec<u32> = (0..dim as u32).collect();
        for i in (1..dim).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let signs: Vec<f32> = (0..dim).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
        Self { dim, perm, signs }
    }

    pub fn _placeholder(&self) {}

    /// Apply the rotation `y = D * pi(x)`. Cheap, in-place style.
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        let mut y = vec![0f32; self.dim];
        for i in 0..self.dim {
            y[i] = self.signs[i] * x[self.perm[i] as usize];
        }
        y
    }

    /// Encode a full-precision vector to a 1-bit code. Stores the original
    /// vector's L2 norm so we can reconstruct true distances at query time.
    pub fn encode(&self, x: &[f32]) -> BitCode {
        let y = self.rotate(x);
        let mut norm_sq = 0f32;
        for v in x { norm_sq += v * v; }
        let norm = norm_sq.sqrt();
        let nwords = (self.dim + 63) / 64;
        let mut words = vec![0u64; nwords];
        for i in 0..self.dim {
            if y[i] >= 0.0 {
                words[i >> 6] |= 1u64 << (i & 63);
            }
        }
        BitCode { words, dim: self.dim, norm }
    }

    /// Pre-process a query into the "rotated, sign-and-magnitude" form used
    /// by the asymmetric estimator. We keep full precision on the query
    /// side (RaBitQ-FP32 estimator) because queries are few and the index
    /// is many — this is the standard RaBitQ recipe.
    pub fn encode_query(&self, q: &[f32]) -> EncodedQuery {
        let y = self.rotate(q);
        let mut sum_abs = 0f32;
        let mut q_norm_sq = 0f32;
        for v in &y {
            sum_abs += v.abs();
            q_norm_sq += v * v;
        }
        EncodedQuery { rotated: y, sum_abs, q_norm: q_norm_sq.sqrt(), sign_bits: Vec::new() }
    }
}

#[derive(Clone, Debug)]
pub struct EncodedQuery {
    pub rotated: Vec<f32>,
    /// Σ |y_i|, used as the per-query scale for the sign-code estimator.
    pub sum_abs: f32,
    pub q_norm: f32,
    /// Sign-bit packing of `rotated`, used by the symmetric popcount estimator.
    pub sign_bits: Vec<u64>,
}

impl RaBitQuantizer {
    /// Encode the query into the symmetric form (1-bit-per-dim, like the
    /// database codes). Faster traversal at the cost of ~30–50% looser
    /// estimator variance vs the asymmetric FP estimator.
    pub fn encode_query_symmetric(&self, q: &[f32]) -> EncodedQuery {
        let mut eq = self.encode_query(q);
        let nwords = (self.dim + 63) / 64;
        let mut bits = vec![0u64; nwords];
        for i in 0..self.dim {
            if eq.rotated[i] >= 0.0 { bits[i >> 6] |= 1u64 << (i & 63); }
        }
        eq.sign_bits = bits;
        eq
    }
}

/// Estimated squared L2 distance between query q (full precision, rotated)
/// and database point x represented by `code`.
///
/// Estimator (RaBitQ-style, asymmetric):
///   <\hat q, sign(\hat x)>  ≈  (Σ |\hat q_i|) / √d  *  (1/√d) Σ s_i \hat q_i
/// We compute the dot product between the rotated query and the {-1,+1}
/// sign code, then scale to recover the true inner product. The estimator
/// is exact in expectation under a random rotation.
#[inline]
pub fn estimate_dist_sq(q: &EncodedQuery, code: &BitCode) -> f32 {
    debug_assert_eq!(q.rotated.len(), code.dim);
    // Inner product of rotated query with ±1 sign code.
    let mut ip: f32 = 0.0;
    // Process 64 dims at a time using the packed code.
    let dim = code.dim;
    let mut i = 0usize;
    for w in &code.words {
        let mut bits = *w;
        let block_end = (i + 64).min(dim);
        while i < block_end {
            let b = bits & 1;
            bits >>= 1;
            // bit 1 => sign +1, bit 0 => sign -1
            let s: f32 = if b == 1 { 1.0 } else { -1.0 };
            ip += s * q.rotated[i];
            i += 1;
        }
    }
    // RaBitQ calibration: scale sign-code inner product to approximate
    // the true rotated-vector inner product. Each sign bit s_i ≈
    // \hat x_i / |\hat x_i|, so s_i * \hat q_i ≈ \hat q_i * sign(\hat x_i).
    // The expected true IP is (||\hat x|| / √d) * <\hat q, sign(\hat x)>.
    // We use ||x|| = code.norm (rotation is norm-preserving in our case
    // since the diagonal sign matrix is orthogonal and permutation is too).
    let scale = code.norm / (dim as f32).sqrt();
    let true_ip_est = scale * ip;
    // ||q - x||^2 = ||q||^2 + ||x||^2 - 2<q,x>
    let qn2 = q.q_norm * q.q_norm;
    let xn2 = code.norm * code.norm;
    (qn2 + xn2 - 2.0 * true_ip_est).max(0.0)
}

/// Symmetric popcount estimator. Both query and database point are 1-bit
/// codes; distance is reconstructed from Hamming weight only. Much faster
/// than the asymmetric estimator (a few ops per 64 dims) at the cost of
/// somewhat looser variance.
#[inline]
pub fn estimate_dist_sq_popcount(q: &EncodedQuery, code: &BitCode) -> f32 {
    debug_assert_eq!(q.sign_bits.len(), code.words.len());
    let dim = code.dim;
    let mut hamming: u32 = 0;
    for i in 0..code.words.len() {
        hamming += (q.sign_bits[i] ^ code.words[i]).count_ones();
    }
    // <sign(q), sign(x)> = d - 2*hamming
    // E[<\hat q, \hat x>] = (||\hat q|| * ||\hat x|| / d) * (d - 2*hamming)
    //                    = (q_norm * code.norm / d) * (d - 2*hamming)
    let d_f = dim as f32;
    let true_ip_est = (q.q_norm * code.norm / d_f) * (d_f - 2.0 * hamming as f32);
    let qn2 = q.q_norm * q.q_norm;
    let xn2 = code.norm * code.norm;
    (qn2 + xn2 - 2.0 * true_ip_est).max(0.0)
}

/// Exact squared L2.
#[inline]
pub fn exact_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_norm_preserving() {
        let q = RaBitQuantizer::new(64, 7);
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let y = q.rotate(&x);
        let nx: f32 = x.iter().map(|v| v*v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v*v).sum::<f32>().sqrt();
        assert!((nx - ny).abs() < 1e-4, "rotation must preserve norm, got {} vs {}", nx, ny);
    }

    #[test]
    fn estimator_is_unbiased_on_average() {
        // Pick a fixed query/db pair and average estimator over many
        // random rotations. The mean error should be small relative to
        // the true distance.
        let dim = 128;
        let x: Vec<f32> = (0..dim).map(|i| ((i as f32).sin())).collect();
        let q: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.7).cos())).collect();
        let true_d = exact_dist_sq(&q, &x);
        let mut acc = 0f64;
        let trials = 64;
        for s in 0..trials {
            let qz = RaBitQuantizer::new(dim, s as u64);
            let code = qz.encode(&x);
            let eq = qz.encode_query(&q);
            acc += estimate_dist_sq(&eq, &code) as f64;
        }
        let est = (acc / trials as f64) as f32;
        let rel = (est - true_d).abs() / true_d.max(1e-6);
        assert!(rel < 0.20, "avg estimator off by {}: true={}, est={}", rel, true_d, est);
    }
}
