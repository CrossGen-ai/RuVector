//! Pyramid Vector Quantization (PVQ) — spherical lattice codes for unit-norm ANN.
//!
//! Given an L2-unit vector x ∈ ℝ^d, PVQ represents x by the integer vector
//! y ∈ ℤ^d with Σ|y_i| = K that minimizes ||x - y/||y||_2||. The decoded
//! approximation is x̂ = y / ||y||_2 — a point on the discrete cross-polytope
//! S(d, K) projected to the unit sphere.
//!
//! Why: training-free, deterministic, closed-form encode (Fischer's search),
//! codebook size grows combinatorially with (d, K) so a single knob tunes
//! quality vs. bytes. Scans on raw int codes avoid LUT builds.
//!
//! Storage per vector: `d` × i8 (1 B/dim) + f32 norm (4 B) = `d + 4` B.
//! For d=128 that's 132 B vs 512 B float32 — ~3.9× smaller.

#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;

/// A trained PVQ codec. Carries only the dimension and L1 budget K.
#[derive(Clone, Debug)]
pub struct PvqCodec {
    pub d: usize,
    pub k: u32,
}

/// Encoded vector. `code` is signed small-integer per dim; `inv_norm` = 1/||code||_2.
#[derive(Clone, Debug)]
pub struct PvqCode {
    pub code: Vec<i8>,
    pub inv_norm: f32,
}

/// Flat PVQ index: a bank of encoded unit vectors + scan.
pub struct PvqIndex {
    pub codec: PvqCodec,
    pub codes: Vec<i8>,       // n × d row-major
    pub inv_norms: Vec<f32>,  // n
    pub n: usize,
}

impl PvqCodec {
    pub fn new(d: usize, k: u32) -> Self {
        assert!(d > 0 && k > 0, "d and k must be positive");
        assert!(k <= 127, "k must fit in i8 range per dim (<=127)");
        Self { d, k }
    }

    /// Bytes stored per vector (code + inverse-norm scalar).
    pub fn bytes_per_vector(&self) -> usize {
        self.d + 4
    }

    /// Encode a unit-L2 vector into a PVQ code.
    ///
    /// Uses the standard Fischer "project to L1 then correct" algorithm:
    ///   1. Normalize to L2 unit if not already (defensive).
    ///   2. Rescale so Σ|x_i| ≈ K: s = K / Σ|x_i|.
    ///   3. Round each s·x_i to nearest int.
    ///   4. Fix integer deficit δ = K − Σ|y_i| by incrementing/decrementing
    ///      the coords whose rounding residuals best improve ||x − y||.
    pub fn encode(&self, x: &[f32]) -> PvqCode {
        assert_eq!(x.len(), self.d);
        let d = self.d;
        let k_target = self.k as i32;

        // Defensive renormalization. PVQ assumes a unit vector; still works
        // for non-unit inputs but with suboptimal MSE.
        let l2: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv_l2 = if l2 > 1e-12 { 1.0 / l2 } else { 1.0 };

        // Step 2 — scale to L1 budget.
        let mut xs: Vec<f32> = x.iter().map(|v| v * inv_l2).collect();
        let abs_sum: f32 = xs.iter().map(|v| v.abs()).sum();
        if abs_sum < 1e-12 {
            // All-zero input: emit canonical first-axis spike.
            let mut code = vec![0_i8; d];
            code[0] = k_target.min(127) as i8;
            let inv_norm = 1.0 / (k_target as f32);
            return PvqCode { code, inv_norm };
        }
        let scale = k_target as f32 / abs_sum;
        for v in xs.iter_mut() {
            *v *= scale;
        }

        // Step 3 — round toward zero (truncate magnitudes) and track residuals.
        let mut y: Vec<i32> = vec![0; d];
        let mut sign: Vec<i8> = vec![1; d];
        // Residual = |scaled x| − |y| (always ≥ 0 by truncation).
        let mut residual: Vec<f32> = vec![0.0; d];
        let mut sum_abs: i32 = 0;
        for i in 0..d {
            sign[i] = if xs[i] < 0.0 { -1 } else { 1 };
            let abs_scaled = xs[i].abs();
            let floor = abs_scaled.floor();
            y[i] = floor as i32;
            residual[i] = abs_scaled - floor;
            sum_abs += y[i];
        }

        // Step 4 — fix deficit δ = K − sum|y|. Positive => need to bump up
        // the coords with largest residual; negative => reduce the smallest.
        let mut delta: i32 = k_target - sum_abs;

        if delta > 0 {
            // Partial sort: pick `delta` coords with highest residual.
            let mut idx: Vec<usize> = (0..d).collect();
            idx.sort_unstable_by(|&a, &b| {
                residual[b]
                    .partial_cmp(&residual[a])
                    .unwrap_or(Ordering::Equal)
            });
            for &i in idx.iter().take(delta as usize) {
                y[i] += 1;
            }
        } else if delta < 0 {
            // Need to subtract; pick coords with smallest residual AND y>0.
            let mut idx: Vec<usize> = (0..d).filter(|&i| y[i] > 0).collect();
            idx.sort_unstable_by(|&a, &b| {
                residual[a]
                    .partial_cmp(&residual[b])
                    .unwrap_or(Ordering::Equal)
            });
            let need = (-delta) as usize;
            let mut done = 0;
            for &i in idx.iter() {
                if done >= need {
                    break;
                }
                y[i] -= 1;
                done += 1;
            }
            // Rare corner: if we still have deficit, repeat (shouldn't happen
            // for sane K ≪ d·max_mag but be defensive).
            delta = k_target - y.iter().sum::<i32>();
            while delta < 0 {
                // Decrement max |y|.
                let i = (0..d).max_by_key(|&i| y[i]).unwrap();
                if y[i] == 0 {
                    break;
                }
                y[i] -= 1;
                delta += 1;
            }
        }

        // Compose signed code and L2 norm of the integer code.
        let mut code = vec![0_i8; d];
        let mut sq: i64 = 0;
        for i in 0..d {
            let v = y[i] as i32 * sign[i] as i32;
            // Clamp defensively (K ≤ 127 by construction).
            let clamped = v.clamp(-127, 127) as i8;
            code[i] = clamped;
            sq += (v as i64) * (v as i64);
        }
        let norm = (sq as f64).sqrt() as f32;
        let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 0.0 };

        PvqCode { code, inv_norm }
    }

    /// Decode to a float approximation (unit-norm).
    pub fn decode(&self, c: &PvqCode) -> Vec<f32> {
        c.code.iter().map(|&v| v as f32 * c.inv_norm).collect()
    }
}

impl PvqIndex {
    /// Build a flat PVQ index from an iterator of row vectors.
    pub fn build(codec: PvqCodec, vectors: &[Vec<f32>]) -> Self {
        let n = vectors.len();
        let d = codec.d;
        let mut codes = Vec::with_capacity(n * d);
        let mut inv_norms = Vec::with_capacity(n);
        for v in vectors {
            let c = codec.encode(v);
            codes.extend_from_slice(&c.code);
            inv_norms.push(c.inv_norm);
        }
        Self {
            codec,
            codes,
            inv_norms,
            n,
        }
    }

    /// Return the `top` nearest neighbors by cosine (= inner product on unit vectors).
    ///
    /// Assumes `query` is unit-L2 (callers must normalize). Scores are
    /// `<q, code_i> * inv_norm_i`, which equals cosine between q and the
    /// decoded unit-norm approximation.
    pub fn search(&self, query: &[f32], top: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.codec.d);
        let d = self.codec.d;
        let mut heap: Vec<(usize, f32)> = Vec::with_capacity(top + 1);
        for i in 0..self.n {
            let row = &self.codes[i * d..(i + 1) * d];
            let mut dot: f32 = 0.0;
            // Scalar dot — kept simple so build/test numbers are honest.
            for j in 0..d {
                dot += query[j] * (row[j] as f32);
            }
            let score = dot * self.inv_norms[i];
            if heap.len() < top {
                heap.push((i, score));
                heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            } else if score > heap[top - 1].1 {
                heap[top - 1] = (i, score);
                heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
        }
        heap
    }

    pub fn bytes_total(&self) -> usize {
        self.codes.len() + self.inv_norms.len() * 4
    }
}

/// L2-normalize a vector in place.
pub fn l2_normalize(x: &mut [f32]) {
    let n: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 1e-12 {
        let inv = 1.0 / n;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// MSE between a vector and its PVQ reconstruction.
pub fn reconstruction_mse(codec: &PvqCodec, x: &[f32]) -> f32 {
    let c = codec.encode(x);
    let x_hat = codec.decode(&c);
    let mut xn = x.to_vec();
    l2_normalize(&mut xn);
    xn.iter()
        .zip(x_hat.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / xn.len() as f32
}

/// Exact (float32) cosine top-K for ground truth. Returns indices only.
pub fn flat_cosine_topk(vectors: &[Vec<f32>], query: &[f32], top: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.into_iter().take(top).map(|(i, _)| i).collect()
}

/// Recall@K between an approximate result and exact ground truth.
pub fn recall_at(exact: &[usize], approx: &[(usize, f32)]) -> f32 {
    let set: std::collections::HashSet<usize> = exact.iter().copied().collect();
    let hits = approx.iter().filter(|(i, _)| set.contains(i)).count();
    hits as f32 / exact.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_unit_vec(d: usize, rng: &mut StdRng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn encode_preserves_l1_budget() {
        let codec = PvqCodec::new(32, 16);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..10 {
            let v = random_unit_vec(32, &mut rng);
            let c = codec.encode(&v);
            let l1: i32 = c.code.iter().map(|&x| (x as i32).abs()).sum();
            assert_eq!(l1, 16, "L1 budget violated");
        }
    }

    #[test]
    fn mse_decreases_with_k() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(7);
        let v = random_unit_vec(d, &mut rng);
        let mse_small = reconstruction_mse(&PvqCodec::new(d, 8), &v);
        let mse_large = reconstruction_mse(&PvqCodec::new(d, 48), &v);
        assert!(
            mse_large < mse_small,
            "larger K should give lower MSE: small={mse_small} large={mse_large}"
        );
    }

    #[test]
    fn index_finds_self() {
        let d = 48;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| random_unit_vec(d, &mut rng)).collect();
        let index = PvqIndex::build(PvqCodec::new(d, 32), &vectors);
        // Query = vector 7; it should be near the top with K=32 (fine codes).
        let q = &vectors[7];
        let hits = index.search(q, 5);
        let hit_ids: Vec<usize> = hits.iter().map(|(i, _)| *i).collect();
        assert!(
            hit_ids.contains(&7),
            "self-query missed: hits={hit_ids:?}"
        );
    }

    #[test]
    fn recall_improves_with_k() {
        let d = 64;
        let n = 400;
        let topk = 10;
        let mut rng = StdRng::seed_from_u64(99);
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vec(d, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..20).map(|_| random_unit_vec(d, &mut rng)).collect();

        let mut recall_small = 0.0;
        let mut recall_large = 0.0;
        let idx_s = PvqIndex::build(PvqCodec::new(d, 8), &vectors);
        let idx_l = PvqIndex::build(PvqCodec::new(d, 48), &vectors);
        for q in &queries {
            let gt = flat_cosine_topk(&vectors, q, topk);
            recall_small += recall_at(&gt, &idx_s.search(q, topk));
            recall_large += recall_at(&gt, &idx_l.search(q, topk));
        }
        recall_small /= queries.len() as f32;
        recall_large /= queries.len() as f32;
        assert!(
            recall_large > recall_small,
            "larger K should improve recall: small={recall_small} large={recall_large}"
        );
    }
}
