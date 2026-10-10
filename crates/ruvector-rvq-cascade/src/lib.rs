//! Residual Vector Quantization (RVQ) cascade rerank.
//!
//! Design: insert a cheap progressive refinement stage between coarse
//! IVF-ADC filtering and full-precision verification. Each candidate is
//! encoded as a sum of K codewords drawn from K sequentially-trained
//! codebooks (c0 + r1 + r2 + ... + r_{K-1}). At query time we build K
//! lookup tables once per query and the per-candidate distance estimate
//! after s stages is just s LUT lookups + s-1 adds.
//!
//! The cascade: start from PQ-estimated distance, refine with RVQ stage
//! 1, prune to top-N1; refine stage 2 on survivors, prune to top-N2;
//! then full-precision verify the remaining shortlist.
//!
//! All code is scalar portable Rust so numbers reproduce bit-exactly on
//! any x86/arm box. Call sites can swap in SIMD later behind the trait.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub type Vec32 = Vec<f32>;

/// A single-level product-free vector quantizer (plain k-means over the
/// whole vector). RVQ stacks K of these on residuals.
#[derive(Debug, Clone)]
pub struct VQStage {
    pub dim: usize,
    pub k: usize,
    pub centroids: Vec<f32>, // k * dim, row-major
}

impl VQStage {
    pub fn train(data: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(dim > 0 && k > 0 && !data.is_empty());
        let n = data.len() / dim;
        assert!(n >= k, "need at least k vectors to train");
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut centroids = vec![0.0f32; k * dim];
        // k-means++ init
        let first = rng.gen_range(0..n);
        centroids[..dim].copy_from_slice(&data[first * dim..(first + 1) * dim]);
        let mut d2 = vec![f32::INFINITY; n];
        for ci in 1..k {
            let base = (ci - 1) * dim;
            let c = &centroids[base..base + dim];
            for i in 0..n {
                let v = &data[i * dim..(i + 1) * dim];
                let d = l2sq(v, c);
                if d < d2[i] {
                    d2[i] = d;
                }
            }
            let total: f64 = d2.iter().map(|&x| x as f64).sum();
            let mut target = rng.gen::<f64>() * total.max(1e-12);
            let mut pick = n - 1;
            for i in 0..n {
                target -= d2[i] as f64;
                if target <= 0.0 {
                    pick = i;
                    break;
                }
            }
            centroids[ci * dim..(ci + 1) * dim]
                .copy_from_slice(&data[pick * dim..(pick + 1) * dim]);
        }
        // Lloyd iterations
        let mut assign = vec![0u32; n];
        for _ in 0..iters {
            // E
            for i in 0..n {
                let v = &data[i * dim..(i + 1) * dim];
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = l2sq(v, &centroids[c * dim..(c + 1) * dim]);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assign[i] = best as u32;
            }
            // M
            let mut sums = vec![0.0f64; k * dim];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let a = assign[i] as usize;
                counts[a] += 1;
                for j in 0..dim {
                    sums[a * dim + j] += data[i * dim + j] as f64;
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f64;
                    for j in 0..dim {
                        centroids[c * dim + j] = (sums[c * dim + j] * inv) as f32;
                    }
                } else {
                    // reseed an empty cluster from a random point
                    let r = rng.gen_range(0..n);
                    centroids[c * dim..(c + 1) * dim]
                        .copy_from_slice(&data[r * dim..(r + 1) * dim]);
                }
            }
        }
        VQStage { dim, k, centroids }
    }

    /// Assign a vector to its nearest centroid.
    pub fn encode(&self, v: &[f32]) -> u8 {
        assert_eq!(v.len(), self.dim);
        assert!(self.k <= 256);
        let mut best = 0u8;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k {
            let d = l2sq(v, &self.centroids[c * self.dim..(c + 1) * self.dim]);
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        best
    }

    /// Dot/L2 LUT for one query vector: lut[c] = l2sq(q, centroid_c).
    pub fn l2_lut(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim);
        let mut out = vec![0.0f32; self.k];
        for c in 0..self.k {
            out[c] = l2sq(q, &self.centroids[c * self.dim..(c + 1) * self.dim]);
        }
        out
    }

    /// Dot-product LUT for one query vector: lut[c] = <q, centroid_c>.
    pub fn dot_lut(&self, q: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.k];
        for c in 0..self.k {
            out[c] = dot(q, &self.centroids[c * self.dim..(c + 1) * self.dim]);
        }
        out
    }

    pub fn centroid(&self, c: u8) -> &[f32] {
        let c = c as usize;
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }
}

/// A stacked Residual Vector Quantizer: K stages, each 8-bit (256 cent).
#[derive(Debug, Clone)]
pub struct RvqCascade {
    pub dim: usize,
    pub stages: Vec<VQStage>,
    /// Per-vector partial recon norms. `partial_norms[s * n + i]` =
    /// ||sum_{s' <= s} c_{s'}[code_{s'}[i]]||^2, so query-time partial
    /// L2 estimates are O(1) in `dim` (just s LUT lookups + 1 norm).
    pub partial_norms: Vec<f32>,
    /// Per-stage, per-vector code: codes[s * n + i].
    pub codes: Vec<u8>,
    pub n: usize,
}

impl RvqCascade {
    /// Train K stages on `data` (n * dim row-major), 256 centroids each.
    pub fn train(data: &[f32], dim: usize, k_stages: usize, iters: usize, seed: u64) -> Self {
        assert!(dim > 0 && k_stages > 0);
        let n = data.len() / dim;
        let mut stages = Vec::with_capacity(k_stages);
        let mut residual = data.to_vec();
        let mut codes = vec![0u8; k_stages * n];
        for s in 0..k_stages {
            let stage = VQStage::train(&residual, dim, 256, iters, seed.wrapping_add(s as u64));
            // encode + update residual
            for i in 0..n {
                let v = &residual[i * dim..(i + 1) * dim];
                let c = stage.encode(v);
                codes[s * n + i] = c;
                let cen = stage.centroid(c);
                for j in 0..dim {
                    residual[i * dim + j] -= cen[j];
                }
            }
            stages.push(stage);
        }
        // Precompute partial recon norms per stage: partial_norms[s*n+i]
        // = ||sum_{s' <= s} c_{s'}[code_{s'}[i]]||^2
        let mut recon = vec![0.0f32; dim];
        let mut partial_norms = vec![0.0f32; k_stages * n];
        for i in 0..n {
            for v in recon.iter_mut() {
                *v = 0.0;
            }
            for s in 0..k_stages {
                let c = codes[s * n + i];
                let cen = stages[s].centroid(c);
                for j in 0..dim {
                    recon[j] += cen[j];
                }
                partial_norms[s * n + i] = recon.iter().map(|&x| x * x).sum();
            }
        }
        RvqCascade {
            dim,
            stages,
            partial_norms,
            codes,
            n,
        }
    }

    /// Build per-stage dot-LUTs for a query.
    pub fn build_dot_luts(&self, q: &[f32]) -> Vec<Vec<f32>> {
        self.stages.iter().map(|s| s.dot_lut(q)).collect()
    }

    /// Estimated L2 distance using stages 0..=last_stage (inclusive).
    /// O(last_stage+1) per candidate: s LUT lookups + 1 norm fetch.
    ///
    /// We use the full-K reconstruction norm `recon_norms[i]` as the
    /// x-norm proxy at every stage. For a well-trained RVQ the stages
    /// after `last_stage` contribute near-zero expected inner product
    /// with the query, so bias is small and shrinks with more stages.
    /// This is the standard "cascading inner-product estimator" trick
    /// in multi-stage ADC pipelines.
    pub fn estimate_l2sq(
        &self,
        i: usize,
        last_stage: usize,
        dot_luts: &[Vec<f32>],
        q_norm_sq: f32,
    ) -> f32 {
        let mut dot_qx = 0.0f32;
        for s in 0..=last_stage {
            let c = self.codes[s * self.n + i] as usize;
            dot_qx += dot_luts[s][c];
        }
        let x_norm_sq = self.partial_norms[last_stage * self.n + i];
        q_norm_sq + x_norm_sq - 2.0 * dot_qx
    }

    /// Fast estimator: uses only dot (ignores norms). Rank-preserving when
    /// candidates all have similar reconstruction norm — fine inside one
    /// IVF cell after norm-bucketing. Returns NEGATED dot so smaller = closer.
    pub fn estimate_fast(&self, i: usize, last_stage: usize, dot_luts: &[Vec<f32>]) -> f32 {
        let mut dot_qx = 0.0f32;
        for s in 0..=last_stage {
            let c = self.codes[s * self.n + i] as usize;
            dot_qx += dot_luts[s][c];
        }
        -dot_qx
    }

    pub fn memory_bytes(&self) -> usize {
        let cb = self.stages.iter().map(|s| s.centroids.len() * 4).sum::<usize>();
        let cd = self.codes.len();
        let rn = self.partial_norms.len() * 4;
        cb + cd + rn
    }
}

/// Baseline: naive 8-bit product quantizer with m subspaces.
#[derive(Debug, Clone)]
pub struct Pq8 {
    pub dim: usize,
    pub m: usize,
    pub sub_dim: usize,
    /// m * 256 * sub_dim
    pub codebooks: Vec<f32>,
    /// m * n
    pub codes: Vec<u8>,
    pub n: usize,
}

impl Pq8 {
    pub fn train(data: &[f32], dim: usize, m: usize, iters: usize, seed: u64) -> Self {
        assert!(dim % m == 0);
        let sub_dim = dim / m;
        let n = data.len() / dim;
        let mut codebooks = vec![0.0f32; m * 256 * sub_dim];
        let mut codes = vec![0u8; m * n];
        for sub in 0..m {
            // extract subspace
            let mut sub_data = vec![0.0f32; n * sub_dim];
            for i in 0..n {
                let off = i * dim + sub * sub_dim;
                sub_data[i * sub_dim..(i + 1) * sub_dim]
                    .copy_from_slice(&data[off..off + sub_dim]);
            }
            let st = VQStage::train(&sub_data, sub_dim, 256, iters, seed.wrapping_add(sub as u64));
            codebooks[sub * 256 * sub_dim..(sub + 1) * 256 * sub_dim]
                .copy_from_slice(&st.centroids);
            for i in 0..n {
                let v = &sub_data[i * sub_dim..(i + 1) * sub_dim];
                codes[sub * n + i] = st.encode(v);
            }
        }
        Pq8 {
            dim,
            m,
            sub_dim,
            codebooks,
            codes,
            n,
        }
    }

    pub fn l2_luts(&self, q: &[f32]) -> Vec<f32> {
        // m * 256
        let mut lut = vec![0.0f32; self.m * 256];
        for sub in 0..self.m {
            let qoff = sub * self.sub_dim;
            let qsub = &q[qoff..qoff + self.sub_dim];
            let cb_base = sub * 256 * self.sub_dim;
            for c in 0..256 {
                let coff = cb_base + c * self.sub_dim;
                let cen = &self.codebooks[coff..coff + self.sub_dim];
                lut[sub * 256 + c] = l2sq(qsub, cen);
            }
        }
        lut
    }

    pub fn adc(&self, i: usize, lut: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for sub in 0..self.m {
            let c = self.codes[sub * self.n + i] as usize;
            s += lut[sub * 256 + c];
        }
        s
    }

    pub fn memory_bytes(&self) -> usize {
        self.codebooks.len() * 4 + self.codes.len()
    }
}

/// Full-precision L2 distance (ground truth / final verify stage).
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Compute recall@k against ground-truth top-k.
pub fn recall_at_k(returned: &[u32], gt: &[u32]) -> f32 {
    let gt_set: std::collections::HashSet<u32> = gt.iter().copied().collect();
    let hit = returned.iter().filter(|x| gt_set.contains(x)).count();
    hit as f32 / gt.len() as f32
}

/// Brute-force top-k by L2 (used to compute ground truth).
pub fn brute_force_topk(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = data.len() / dim;
    let mut dists: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| {
            let v = &data[i as usize * dim..(i as usize + 1) * dim];
            (i, l2sq(q, v))
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}


pub mod prvq;
pub use prvq::{CascadePipeline, Prvq};
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Gaussian-mixture data: 50 clusters with small intra-cluster
        // spread. Realistic for ANN benchmarks (vs degenerate uniform).
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let n_clusters = 50;
        let centers: Vec<f32> = (0..n_clusters * dim)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let c = rng.gen_range(0..n_clusters);
            for j in 0..dim {
                let noise: f32 = rng.gen::<f32>() * 0.3 - 0.15;
                out.push(centers[c * dim + j] + noise);
            }
        }
        out
    }

    #[test]
    fn prvq_beats_pq_at_matched_fetches() {
        // At the same number of full-precision verifies, PRVQ (PQ+RVQ
        // on residual) should produce higher recall than PQ-rerank
        // because its distance estimate is strictly more accurate.
        let (n, dim, nq, k) = (2000usize, 64usize, 32usize, 10usize);
        let data = gen_data(n, dim, 1);
        let queries = gen_data(nq, dim, 2);
        let prvq = Prvq::train(&data, dim, 8, 4, 10, 42);
        let pq = prvq.pq.clone();
        let verify = 50usize;
        let shortlist_pq = 400usize;
        let mut pq_recall = 0.0f32;
        let mut prvq_recall = 0.0f32;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt: Vec<u32> = brute_force_topk(&data, dim, q, k)
                .into_iter()
                .map(|x| x.0)
                .collect();
            // PQ: shortlist=verify, full-verify all
            let lut = pq.l2_luts(q);
            let mut cand: Vec<(u32, f32)> =
                (0..n as u32).map(|i| (i, pq.adc(i as usize, &lut))).collect();
            cand.select_nth_unstable_by(verify - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(verify);
            for p in cand.iter_mut() {
                let v = &data[p.0 as usize * dim..(p.0 as usize + 1) * dim];
                p.1 = l2sq(q, v);
            }
            cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(k);
            let r: Vec<u32> = cand.into_iter().map(|x| x.0).collect();
            pq_recall += recall_at_k(&r, &gt);
            // PRVQ: broader PQ shortlist -> PRVQ prune -> verify
            let dot_luts = prvq.rvq.build_dot_luts(q);
            let mut cand: Vec<(u32, f32)> =
                (0..n as u32).map(|i| (i, pq.adc(i as usize, &lut))).collect();
            cand.select_nth_unstable_by(shortlist_pq - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(shortlist_pq);
            for p in cand.iter_mut() {
                p.1 = prvq.estimate_l2sq(p.0 as usize, &lut, &dot_luts);
            }
            cand.select_nth_unstable_by(verify - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(verify);
            for p in cand.iter_mut() {
                let v = &data[p.0 as usize * dim..(p.0 as usize + 1) * dim];
                p.1 = l2sq(q, v);
            }
            cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(k);
            let r: Vec<u32> = cand.into_iter().map(|x| x.0).collect();
            prvq_recall += recall_at_k(&r, &gt);
        }
        pq_recall /= nq as f32;
        prvq_recall /= nq as f32;
        assert!(
            prvq_recall >= pq_recall,
            "PRVQ recall {} did not beat PQ {} at {} fetches",
            prvq_recall,
            pq_recall,
            verify
        );
    }

    #[test]
    fn rvq_reconstruction_error_shrinks_per_stage() {
        let (n, dim) = (500usize, 32usize);
        let data = gen_data(n, dim, 7);
        let rvq = RvqCascade::train(&data, dim, 3, 10, 9);
        let mut err = vec![0.0f64; rvq.stages.len()];
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let mut recon = vec![0.0f32; dim];
            for s in 0..rvq.stages.len() {
                let c = rvq.codes[s * n + i];
                let cen = rvq.stages[s].centroid(c);
                for j in 0..dim {
                    recon[j] += cen[j];
                }
                err[s] += l2sq(v, &recon) as f64;
            }
        }
        for w in err.windows(2) {
            assert!(w[1] < w[0], "RVQ residual did not shrink: {:?}", err);
        }
    }
}
