//! Rotation-based 1-bit quantizer (RaBitQ-style, simplified).
//!
//! Pipeline:
//!   1. Center the dataset (subtract global mean).
//!   2. Apply a fixed random orthogonal rotation R to spread information across dims.
//!   3. For each rotated vector v, store sign bits sgn(v_i) packed into u64 words.
//!   4. Cache the rotated-vector L2 norm so we can reconstruct an estimated dot product.
//!
//! Distance estimate between query q and stored vector x (after rotation r_q, r_x):
//!   <r_q, r_x> ≈ ||r_q||_1-like dot via popcount on sign-aligned bits.
//!
//! We use the closed form:
//!   d_est(q, x)^2 ≈ ||r_q||^2 + ||r_x||^2 - 2 * <r_q, b_x> * scale_x
//! where b_x is the {-1,+1} sign vector and scale_x compensates for the unit-norm
//! projection. <r_q, b_x> is computed by signing the bits of r_q and using popcount
//! to count agreements: agreements - disagreements = D - 2 * popcount(sign(r_q) XOR b_x).
//!
//! This is the canonical RaBitQ trick. Accuracy is enough as a *first-pass* filter;
//! the symphony index always re-ranks the top-k candidates with exact L2 to recover
//! recall.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Packed 1-bit code for a single vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedCode {
    /// Sign bits of the rotated vector, packed little-endian into u64 words.
    pub bits: Vec<u64>,
    /// L2 norm of the rotated vector (== L2 norm of the centered vector, R is orthogonal).
    pub norm: f32,
}

#[derive(Clone, Debug)]
pub struct RotatedQuantizer {
    dim: usize,
    /// Row-major dim x dim orthogonal rotation. We use a sparse-ish Hadamard-style
    /// random sign matrix combined with a permutation as a practical stand-in. It is
    /// reasonably close to orthogonal in expectation and far cheaper than a full
    /// QR-rotated dense matrix while preserving the RaBitQ accuracy story for
    /// dimensions in the 32..512 range used by the benchmark suite.
    rotation: Vec<f32>,
    mean: Vec<f32>,
    words_per_code: usize,
}

impl RotatedQuantizer {
    pub fn new(dim: usize, seed: u64) -> Self {
        // Build an orthogonal rotation via Gram-Schmidt over a random Gaussian matrix.
        // This is O(D^3); acceptable because D is small (<= 256 in the benchmark).
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut m: Vec<Vec<f32>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        // Box-Muller via two uniforms.
                        let u1: f32 = rng.gen_range(1e-7_f32..1.0);
                        let u2: f32 = rng.gen_range(0.0_f32..1.0);
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                    })
                    .collect()
            })
            .collect();
        // Modified Gram-Schmidt
        for i in 0..dim {
            // normalize row i
            let n = m[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for j in 0..dim {
                m[i][j] /= n;
            }
            // remove projection from subsequent rows
            for k in (i + 1)..dim {
                let dot: f32 = (0..dim).map(|j| m[i][j] * m[k][j]).sum();
                for j in 0..dim {
                    m[k][j] -= dot * m[i][j];
                }
            }
        }
        let mut rotation = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                rotation[i * dim + j] = m[i][j];
            }
        }
        let words_per_code = (dim + 63) / 64;
        Self {
            dim,
            rotation,
            mean: vec![0.0; dim],
            words_per_code,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn words_per_code(&self) -> usize {
        self.words_per_code
    }

    /// Fit the centroid (mean) from a slice of training vectors.
    pub fn fit_mean(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len().max(1);
        let mut acc = vec![0.0_f64; self.dim];
        for v in vectors {
            for (a, &x) in acc.iter_mut().zip(v.iter()) {
                *a += x as f64;
            }
        }
        for (m, a) in self.mean.iter_mut().zip(acc.iter()) {
            *m = (*a / n as f64) as f32;
        }
    }

    /// Rotate a centered vector r = R * (x - mean).
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let mut centered = vec![0.0_f32; self.dim];
        for i in 0..self.dim {
            centered[i] = x[i] - self.mean[i];
        }
        for i in 0..self.dim {
            let row = &self.rotation[i * self.dim..(i + 1) * self.dim];
            let mut s = 0.0_f32;
            for j in 0..self.dim {
                s += row[j] * centered[j];
            }
            out[i] = s;
        }
    }

    /// Encode a single vector into a packed 1-bit code.
    pub fn encode(&self, x: &[f32]) -> QuantizedCode {
        let mut r = vec![0.0_f32; self.dim];
        self.rotate(x, &mut r);
        let mut bits = vec![0u64; self.words_per_code];
        let mut sumsq = 0.0_f32;
        for i in 0..self.dim {
            sumsq += r[i] * r[i];
            if r[i] >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        QuantizedCode {
            bits,
            norm: sumsq.sqrt(),
        }
    }

    /// Pre-process a query: rotate it and produce both a packed-sign vector and the
    /// rotated float vector (the latter is consumed by the distance estimator).
    pub fn prepare_query(&self, q: &[f32]) -> PreparedQuery {
        let mut r = vec![0.0_f32; self.dim];
        self.rotate(q, &mut r);
        let mut bits = vec![0u64; self.words_per_code];
        let mut sumsq = 0.0_f32;
        let mut abs_sum = 0.0_f32;
        for i in 0..self.dim {
            sumsq += r[i] * r[i];
            abs_sum += r[i].abs();
            if r[i] >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        PreparedQuery {
            rotated: r,
            sign_bits: bits,
            norm_sq: sumsq,
            abs_sum,
        }
    }

    /// Estimated squared L2 distance between query and stored quantized code.
    ///
    /// Uses the {-1,+1} sign-Hamming approximation:
    ///   agreements = D - 2 * popcount(q_bits XOR x_bits)
    ///   <sign(r_q), sign(r_x)> = agreements
    /// The inner product <r_q, r_x> is approximated by `(|r_q|_1 / D) * agreements * norm_x`,
    /// which is the standard RaBitQ first-order estimator (cheap, biased low at high D
    /// but monotonic enough for first-pass ordering).
    pub fn estimate_l2_sq(&self, q: &PreparedQuery, code: &QuantizedCode) -> f32 {
        let d = self.dim as f32;
        let mut hamming = 0u32;
        for w in 0..self.words_per_code {
            hamming += (q.sign_bits[w] ^ code.bits[w]).count_ones();
        }
        // For the last word, mask out bits beyond D so they don't pollute the count.
        let valid_bits = self.dim;
        let total_bits = self.words_per_code * 64;
        let pad = total_bits - valid_bits;
        // We did not mask bits in encode (they default to 0), so q has 0s at pad
        // positions and code has 0s too; XOR is 0; no correction needed.
        let _ = pad;
        let agreements = d - 2.0 * hamming as f32;
        // Approximate <r_q, r_x> ~= (q.abs_sum / d) * agreements * (code.norm / |r_x|_1 estimate).
        // Use abs_sum / d as the average magnitude proxy on the query side; on the
        // stored side we have code.norm and an implicit average magnitude ~ norm / sqrt(d).
        let q_mag = q.abs_sum / d;
        let x_mag = code.norm / (d.sqrt().max(1e-6));
        let inner_est = q_mag * x_mag * agreements;
        // d_est^2 = ||q||^2 + ||x||^2 - 2 * <q, x>
        (q.norm_sq + code.norm * code.norm - 2.0 * inner_est).max(0.0)
    }
}

/// Pre-processed query state, reusable across all codes during a single search.
pub struct PreparedQuery {
    pub rotated: Vec<f32>,
    pub sign_bits: Vec<u64>,
    pub norm_sq: f32,
    pub abs_sum: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthogonal() {
        let q = RotatedQuantizer::new(16, 42);
        // R * R^T == I (within tolerance).
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0_f32;
                for k in 0..16 {
                    s += q.rotation[i * 16 + k] * q.rotation[j * 16 + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - expected).abs() < 1e-3,
                    "R*R^T[{i},{j}] = {s}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn estimator_orders_correctly() {
        // Build a tiny dataset where two vectors are very close and one is far.
        let q = RotatedQuantizer::new(64, 7);
        let near_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.01).sin()).collect();
        let near_b: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.01).sin() + 0.001)
            .collect();
        let far: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).cos()).collect();
        let code_near = q.encode(&near_b);
        let code_far = q.encode(&far);
        let pq = q.prepare_query(&near_a);
        let d_near = q.estimate_l2_sq(&pq, &code_near);
        let d_far = q.estimate_l2_sq(&pq, &code_far);
        assert!(d_near < d_far, "expected near < far, got {d_near} vs {d_far}");
    }
}
