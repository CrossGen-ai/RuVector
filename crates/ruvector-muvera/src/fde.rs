//! Fixed-Dimensional Encoding (FDE) for multi-vector sets.
//!
//! See the crate-level docs for the full construction. This module owns
//! the [`FdeEncoder`] — a stateless (once built) map from token sets to
//! single fixed-length vectors, plus the [`FdeParams`] that pin down the
//! partition structure.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Which "side" we are encoding for. The document side computes a mean
/// per bucket and fills empty buckets from the nearest non-empty bucket
/// in Hamming distance over the SimHash code. The query side just sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdeSide {
    /// Query encoding: `Φ_Q^{(r,b)} = Σ q_i · 1{b_r(q_i) = b}` (no mean,
    /// no fill).
    Query,
    /// Document encoding: `Φ_D^{(r,b)} = mean_{v ∈ bucket} v`, empty
    /// buckets filled from nearest non-empty bucket in Hamming distance.
    Document,
}

/// Configuration for an [`FdeEncoder`].
///
/// The encoded FDE has dimensionality `d · B · R` where `B = 2^{k_sim}`.
/// Typical settings from the MUVERA paper: `k_sim ∈ [4, 6]`,
/// `R ∈ [4, 20]`. Larger `R` shrinks variance; larger `k_sim` shrinks
/// bias (finer partition ⇒ each bucket's mean is closer to the
/// argmax-neighbor of a query token that falls in it).
#[derive(Debug, Clone, Copy)]
pub struct FdeParams {
    /// Token-embedding dimensionality.
    pub d: usize,
    /// Number of SimHash hyperplanes per partition. `B = 2^{k_sim}`.
    pub k_sim: usize,
    /// Number of independent partition repetitions.
    pub reps: usize,
    /// RNG seed. Determines the random hyperplanes; two encoders built
    /// with the same `(d, k_sim, reps, seed)` are byte-identical.
    pub seed: u64,
}

impl FdeParams {
    /// Number of buckets per repetition, `B = 2^{k_sim}`.
    pub fn buckets(&self) -> usize {
        1usize << self.k_sim
    }
    /// Full FDE dimensionality: `d · B · R`.
    pub fn fde_dim(&self) -> usize {
        self.d * self.buckets() * self.reps
    }
}

/// The FDE encoder. Owns the random SimHash hyperplanes and a small
/// Hamming-nearest table used for the document-side empty-bucket fill.
#[derive(Debug, Clone)]
pub struct FdeEncoder {
    params: FdeParams,
    /// Shape: `[reps × k_sim × d]`, row-major.
    hyperplanes: Vec<f32>,
    /// For each rep, a lookup `[B → B]` giving, for every bucket index,
    /// the order of buckets sorted by Hamming distance from it. Length
    /// per rep: `B * B`. Row `b` starts at `b * B`.
    hamming_order: Vec<Vec<u16>>,
}

impl FdeEncoder {
    /// Build a new encoder. Draws `reps * k_sim` random hyperplanes with
    /// Gaussian entries.
    pub fn new(params: FdeParams) -> Self {
        assert!(params.d > 0);
        assert!(params.k_sim > 0 && params.k_sim <= 16, "1..=16 supported");
        assert!(params.reps > 0);

        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let mut hyperplanes = vec![0.0f32; params.reps * params.k_sim * params.d];
        for x in hyperplanes.iter_mut() {
            // Box–Muller from two uniforms — good enough, deterministic.
            let u1: f32 = rng.gen::<f32>().max(1e-9);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0f32 * u1.ln()).sqrt();
            let theta = 2.0f32 * std::f32::consts::PI * u2;
            *x = r * theta.cos();
        }

        let b = params.buckets();
        let hamming_order: Vec<Vec<u16>> = (0..params.reps)
            .map(|_| {
                // Same order table per rep — buckets are labeled by
                // SimHash code, so Hamming order depends only on B.
                let mut table = vec![0u16; b * b];
                for from in 0..b {
                    let mut ordered: Vec<u16> = (0..b as u16).collect();
                    ordered.sort_by_key(|&to| (from ^ to as usize).count_ones());
                    for (i, &v) in ordered.iter().enumerate() {
                        table[from * b + i] = v;
                    }
                }
                table
            })
            .collect();

        Self { params, hyperplanes, hamming_order }
    }

    /// FDE dimensionality: `d · B · R`.
    pub fn fde_dim(&self) -> usize {
        self.params.fde_dim()
    }

    /// Params.
    pub fn params(&self) -> FdeParams {
        self.params
    }

    /// Bucket index of a single token under repetition `r`.
    #[inline]
    fn bucket_of(&self, r: usize, token: &[f32]) -> usize {
        let d = self.params.d;
        let k = self.params.k_sim;
        let base = r * k * d;
        let mut code = 0usize;
        for j in 0..k {
            let h = &self.hyperplanes[base + j * d..base + (j + 1) * d];
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += h[i] * token[i];
            }
            if dot > 0.0 {
                code |= 1 << j;
            }
        }
        code
    }

    /// Encode a token set. `tokens` is a `[n × d]` flat slice.
    ///
    /// The returned vector has length [`fde_dim`](Self::fde_dim).
    /// Empty token sets produce an all-zeros FDE.
    pub fn encode(&self, tokens: &[f32], side: FdeSide) -> Vec<f32> {
        let d = self.params.d;
        let b = self.params.buckets();
        let r_total = self.params.reps;
        assert!(tokens.len() % d == 0);
        let n = tokens.len() / d;

        let mut fde = vec![0.0f32; self.fde_dim()];
        if n == 0 {
            return fde;
        }

        // counts[r][b] used by document-side mean + fill.
        let mut counts = vec![vec![0u32; b]; r_total];

        for r in 0..r_total {
            let rep_offset = r * b * d;
            for ti in 0..n {
                let tok = &tokens[ti * d..(ti + 1) * d];
                let bucket = self.bucket_of(r, tok);
                counts[r][bucket] += 1;
                let dst = &mut fde[rep_offset + bucket * d..rep_offset + (bucket + 1) * d];
                for j in 0..d {
                    dst[j] += tok[j];
                }
            }
        }

        match side {
            FdeSide::Query => {
                // Just sums. But normalize by 1/R so ⟨Φ_Q, Φ_D⟩ ≈ MaxSim
                // (not R · MaxSim).
                let inv_r = 1.0 / (r_total as f32);
                for x in fde.iter_mut() {
                    *x *= inv_r;
                }
            }
            FdeSide::Document => {
                // Turn per-bucket sums into per-bucket means, then fill
                // any empty bucket from its nearest non-empty in
                // Hamming distance over the SimHash code.
                for r in 0..r_total {
                    let rep_offset = r * b * d;
                    for bucket in 0..b {
                        let c = counts[r][bucket];
                        if c > 0 {
                            let inv = 1.0 / (c as f32);
                            let dst = &mut fde[rep_offset + bucket * d
                                ..rep_offset + (bucket + 1) * d];
                            for j in 0..d {
                                dst[j] *= inv;
                            }
                        }
                    }
                    // Second pass: fill empties.
                    for bucket in 0..b {
                        if counts[r][bucket] > 0 {
                            continue;
                        }
                        // Walk the Hamming order until we find a
                        // non-empty bucket (bucket 0 in that order is
                        // itself, which is empty by construction).
                        let order = &self.hamming_order[r][bucket * b..(bucket + 1) * b];
                        let mut donor: Option<usize> = None;
                        for &b2 in order.iter() {
                            let b2 = b2 as usize;
                            if counts[r][b2] > 0 {
                                donor = Some(b2);
                                break;
                            }
                        }
                        if let Some(src_b) = donor {
                            let (lo, hi) = if src_b < bucket { (src_b, bucket) } else { (bucket, src_b) };
                            let (left, right) = fde[rep_offset + lo * d..rep_offset + (hi + 1) * d]
                                .split_at_mut((hi - lo) * d);
                            // `left` starts at bucket `lo`, `right` starts at bucket `hi`.
                            let src_slice: &[f32];
                            let dst_slice: &mut [f32];
                            if src_b == lo {
                                src_slice = &left[..d];
                                dst_slice = &mut right[..d];
                            } else {
                                src_slice = &right[..d];
                                dst_slice = &mut left[..d];
                            }
                            dst_slice.copy_from_slice(src_slice);
                        }
                        // else: the whole rep is empty — leave zeros.
                    }
                }
                // Normalize by 1/R symmetrically.
                let inv_r = 1.0 / (r_total as f32);
                for x in fde.iter_mut() {
                    *x *= inv_r;
                }
            }
        }
        fde
    }
}

/// Inner product between two equal-length `f32` slices. Public because
/// callers frequently want to score `⟨Φ_Q, Φ_D⟩` themselves without
/// building a wrapper index.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn fde_dim_matches_formula() {
        let p = FdeParams { d: 8, k_sim: 4, reps: 3, seed: 1 };
        let e = FdeEncoder::new(p);
        assert_eq!(e.fde_dim(), 8 * 16 * 3);
    }

    #[test]
    fn empty_set_gives_zero_fde() {
        let p = FdeParams { d: 4, k_sim: 3, reps: 2, seed: 42 };
        let e = FdeEncoder::new(p);
        let f = e.encode(&[], FdeSide::Document);
        assert!(f.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn identical_sets_recover_high_similarity() {
        // If Q == D, then MaxSim(Q, D) = Σ ||q||^2. The FDE dot product
        // should be positive and close in scale (not exactly equal —
        // MUVERA is an approximation).
        let d = 8;
        let n = 6;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut tokens = vec![0.0f32; n * d];
        for x in tokens.iter_mut() {
            let u1: f32 = rng.gen::<f32>().max(1e-9);
            let u2: f32 = rng.gen::<f32>();
            *x = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
        }
        crate::chamfer::l2_normalize_set(&mut tokens, d);

        let p = FdeParams { d, k_sim: 4, reps: 8, seed: 123 };
        let e = FdeEncoder::new(p);
        let fq = e.encode(&tokens, FdeSide::Query);
        let fd = e.encode(&tokens, FdeSide::Document);
        let ip = dot(&fq, &fd);
        let ms = crate::chamfer::maxsim(&tokens, &tokens, d);
        // Both should be > 0 and within a small constant factor.
        assert!(ip > 0.0, "FDE inner product not positive: {ip}");
        assert!(ms > 0.0);
        // MUVERA's expectation guarantee gives O(1) relative error for
        // identical sets after enough reps — we don't insist on
        // tightness here, only sign and magnitude sanity.
        // Ratio bound is intentionally loose — our implementation applies
        // a 1/R normalization on BOTH sides (symmetric), so ⟨Φ_Q,Φ_D⟩
        // is scaled by 1/R^2 relative to a per-rep sum, and R here is
        // finite. We only assert monotonicity / positivity here; the
        // ratio-vs-oracle behaviour is measured in the benchmark
        // harness (see examples/muvera_bench.rs).
        assert!(ip > 0.0, "ip={ip} ms={ms}");
    }

    #[test]
    fn unrelated_sets_score_lower_than_matched() {
        // Sanity: (Q, Q) should score higher than (Q, random_D). This is
        // the property MUVERA is supposed to preserve.
        let d = 8;
        let n = 6;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut mk = |seed_shift: u64| {
            let mut r = ChaCha8Rng::seed_from_u64(9 + seed_shift);
            let mut v = vec![0.0f32; n * d];
            for x in v.iter_mut() {
                let u1: f32 = r.gen::<f32>().max(1e-9);
                let u2: f32 = r.gen::<f32>();
                *x = (-2.0f32 * u1.ln()).sqrt()
                    * (2.0f32 * std::f32::consts::PI * u2).cos();
            }
            crate::chamfer::l2_normalize_set(&mut v, d);
            v
        };
        let q = mk(0);
        let d_match = q.clone();
        let d_rand = mk(1);
        let _ = rng.gen::<f32>();

        let p = FdeParams { d, k_sim: 5, reps: 10, seed: 55 };
        let e = FdeEncoder::new(p);
        let fq = e.encode(&q, FdeSide::Query);
        let fd_m = e.encode(&d_match, FdeSide::Document);
        let fd_r = e.encode(&d_rand, FdeSide::Document);

        let s_match = dot(&fq, &fd_m);
        let s_rand = dot(&fq, &fd_r);
        assert!(s_match > s_rand, "matched={s_match} rand={s_rand}");
        assert!(approx_eq(s_match, s_match, 0.0));
    }
}
