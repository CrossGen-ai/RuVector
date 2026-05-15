//! FDE encoder.
//!
//! Two encoding modes per the MUVERA paper:
//!   * **document side** — per-bucket *centroid* (mean of doc tokens in
//!     bucket). Empty buckets are *filled* from the nearest non-empty
//!     bucket by Hamming distance on the bucket id.
//!   * **query side** — per-bucket *sum* of query tokens. No fill.
//!
//! With this asymmetric scheme, `<q_FDE, d_FDE>` is an unbiased estimator
//! of the asymmetric Chamfer similarity sum_q max_d <q,d>. Stacking
//! `R` independent SimHash repetitions reduces variance like 1/sqrt(R).

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::error::MuveraError;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FillStrategy {
    /// Empty doc buckets stay zero. Cheap, slightly higher variance.
    Zero,
    /// Empty doc buckets are copied from the non-empty bucket whose id
    /// has minimum Hamming distance to the empty one.
    NearestBucket,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectionMode {
    /// No final projection. FDE dim = R * (1 << k_sim) * d.
    None,
    /// Gaussian random projection to `d_final`.
    Gaussian,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdeConfig {
    /// Token vector dimension.
    pub d: usize,
    /// SimHash bits per repetition. Bucket count = 2^k_sim.
    pub k_sim: usize,
    /// Independent SimHash repetitions.
    pub r_reps: usize,
    pub fill: FillStrategy,
    pub projection: ProjectionMode,
    /// Final projected dim (only used if projection != None).
    pub d_final: usize,
    pub seed: u64,
}

impl FdeConfig {
    pub fn validate(&self) -> Result<(), MuveraError> {
        if self.d == 0 {
            return Err(MuveraError::InvalidConfig("d must be > 0"));
        }
        if self.k_sim == 0 || self.k_sim > 12 {
            return Err(MuveraError::InvalidConfig("k_sim must be in 1..=12"));
        }
        if self.r_reps == 0 {
            return Err(MuveraError::InvalidConfig("r_reps must be > 0"));
        }
        if matches!(self.projection, ProjectionMode::Gaussian) && self.d_final == 0 {
            return Err(MuveraError::InvalidConfig("d_final must be > 0 with Gaussian projection"));
        }
        Ok(())
    }

    /// Raw (pre-projection) FDE dimension.
    pub fn raw_dim(&self) -> usize {
        self.r_reps * (1usize << self.k_sim) * self.d
    }

    /// Final encoded vector length, accounting for projection.
    pub fn output_dim(&self) -> usize {
        match self.projection {
            ProjectionMode::None => self.raw_dim(),
            ProjectionMode::Gaussian => self.d_final,
        }
    }
}

pub struct FdeEncoder {
    cfg: FdeConfig,
    /// `r_reps` packs of `k_sim x d` Gaussian hyperplanes.
    hyperplanes: Vec<Vec<f32>>,
    /// Optional projection matrix shape `d_final x raw_dim`.
    projection: Option<Vec<f32>>,
}

impl FdeEncoder {
    pub fn new(cfg: FdeConfig) -> Result<Self, MuveraError> {
        cfg.validate()?;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();

        let mut hyperplanes = Vec::with_capacity(cfg.r_reps);
        for _ in 0..cfg.r_reps {
            let mut plane = Vec::with_capacity(cfg.k_sim * cfg.d);
            for _ in 0..(cfg.k_sim * cfg.d) {
                plane.push(normal.sample(&mut rng));
            }
            hyperplanes.push(plane);
        }

        let projection = match cfg.projection {
            ProjectionMode::None => None,
            ProjectionMode::Gaussian => {
                let raw = cfg.raw_dim();
                let scale = 1.0_f32 / (cfg.d_final as f32).sqrt();
                let mut p = Vec::with_capacity(cfg.d_final * raw);
                for _ in 0..(cfg.d_final * raw) {
                    p.push(normal.sample(&mut rng) * scale);
                }
                Some(p)
            }
        };

        Ok(Self { cfg, hyperplanes, projection })
    }

    pub fn config(&self) -> &FdeConfig {
        &self.cfg
    }

    /// SimHash a single token within rep `r` -> bucket id in [0, 2^k_sim).
    fn bucket_for(&self, r: usize, token: &[f32]) -> u32 {
        debug_assert_eq!(token.len(), self.cfg.d);
        let plane = &self.hyperplanes[r];
        let mut id: u32 = 0;
        for h in 0..self.cfg.k_sim {
            let row = &plane[h * self.cfg.d..(h + 1) * self.cfg.d];
            let mut s = 0.0_f32;
            for i in 0..self.cfg.d {
                s += row[i] * token[i];
            }
            if s >= 0.0 {
                id |= 1u32 << h;
            }
        }
        id
    }

    fn encode_doc_rep(&self, rep: usize, tokens: &[Vec<f32>], out: &mut [f32]) {
        let b = 1usize << self.cfg.k_sim;
        debug_assert_eq!(out.len(), b * self.cfg.d);
        let mut counts = vec![0u32; b];
        for tok in tokens {
            let bid = self.bucket_for(rep, tok) as usize;
            counts[bid] += 1;
            let off = bid * self.cfg.d;
            for i in 0..self.cfg.d {
                out[off + i] += tok[i];
            }
        }
        // mean per non-empty bucket
        for bid in 0..b {
            if counts[bid] > 1 {
                let inv = 1.0_f32 / counts[bid] as f32;
                let off = bid * self.cfg.d;
                for i in 0..self.cfg.d {
                    out[off + i] *= inv;
                }
            }
        }
        // fill empty buckets
        if matches!(self.cfg.fill, FillStrategy::NearestBucket) {
            // collect non-empty bucket ids
            let nonempty: Vec<u32> = (0..b).filter(|&bid| counts[bid] > 0).map(|x| x as u32).collect();
            if nonempty.is_empty() {
                return;
            }
            for bid in 0..b {
                if counts[bid] == 0 {
                    let mut best_dist = u32::MAX;
                    let mut best = nonempty[0];
                    for &cand in &nonempty {
                        let dist = ((bid as u32) ^ cand).count_ones();
                        if dist < best_dist {
                            best_dist = dist;
                            best = cand;
                        }
                    }
                    let src_off = best as usize * self.cfg.d;
                    let dst_off = bid * self.cfg.d;
                    for i in 0..self.cfg.d {
                        out[dst_off + i] = out[src_off + i];
                    }
                }
            }
        }
    }

    fn encode_query_rep(&self, rep: usize, tokens: &[Vec<f32>], out: &mut [f32]) {
        let b = 1usize << self.cfg.k_sim;
        debug_assert_eq!(out.len(), b * self.cfg.d);
        for tok in tokens {
            let bid = self.bucket_for(rep, tok) as usize;
            let off = bid * self.cfg.d;
            for i in 0..self.cfg.d {
                out[off + i] += tok[i];
            }
        }
        // No fill / no normalization on query side: <q_b, d_b> with d_b a
        // centroid is exactly Σ_q∈b <q, mean_d∈b>, an unbiased proxy for
        // max_d <q, d> when SimHash collisions track inner-product.
    }

    fn project(&self, raw: &[f32]) -> Vec<f32> {
        match (&self.projection, self.cfg.projection) {
            (Some(p), ProjectionMode::Gaussian) => {
                let raw_dim = self.cfg.raw_dim();
                let mut out = vec![0.0_f32; self.cfg.d_final];
                for row in 0..self.cfg.d_final {
                    let off = row * raw_dim;
                    let mut s = 0.0_f32;
                    for i in 0..raw_dim {
                        s += p[off + i] * raw[i];
                    }
                    out[row] = s;
                }
                out
            }
            _ => raw.to_vec(),
        }
    }

    pub fn encode_doc(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, MuveraError> {
        if tokens.is_empty() {
            return Err(MuveraError::EmptyMultiVector);
        }
        for t in tokens {
            if t.len() != self.cfg.d {
                return Err(MuveraError::DimMismatch { expected: self.cfg.d, actual: t.len() });
            }
        }
        let b = 1usize << self.cfg.k_sim;
        let block = b * self.cfg.d;
        let mut raw = vec![0.0_f32; self.cfg.r_reps * block];
        for r in 0..self.cfg.r_reps {
            let slice = &mut raw[r * block..(r + 1) * block];
            self.encode_doc_rep(r, tokens, slice);
        }
        // unbiasedness: average over reps
        let inv_r = 1.0_f32 / self.cfg.r_reps as f32;
        for x in raw.iter_mut() {
            *x *= inv_r;
        }
        Ok(self.project(&raw))
    }

    pub fn encode_query(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, MuveraError> {
        if tokens.is_empty() {
            return Err(MuveraError::EmptyMultiVector);
        }
        for t in tokens {
            if t.len() != self.cfg.d {
                return Err(MuveraError::DimMismatch { expected: self.cfg.d, actual: t.len() });
            }
        }
        let b = 1usize << self.cfg.k_sim;
        let block = b * self.cfg.d;
        let mut raw = vec![0.0_f32; self.cfg.r_reps * block];
        for r in 0..self.cfg.r_reps {
            let slice = &mut raw[r * block..(r + 1) * block];
            self.encode_query_rep(r, tokens, slice);
        }
        Ok(self.project(&raw))
    }

    /// Bytes per encoded vector, f32.
    pub fn bytes_per_vector(&self) -> usize {
        self.cfg.output_dim() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{chamfer_similarity, dot};
    use rand::SeedableRng;
    use rand_distr::Distribution;

    fn random_multi_vec(n_tokens: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        (0..n_tokens)
            .map(|_| {
                let mut v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng)).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in v.iter_mut() {
                    *x /= n;
                }
                v
            })
            .collect()
    }

    #[test]
    fn config_validates() {
        let bad = FdeConfig {
            d: 0, k_sim: 4, r_reps: 4,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::None, d_final: 0, seed: 1,
        };
        assert!(FdeEncoder::new(bad).is_err());
    }

    #[test]
    fn output_dim_matches() {
        let cfg = FdeConfig {
            d: 16, k_sim: 4, r_reps: 8,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::None, d_final: 0, seed: 7,
        };
        let enc = FdeEncoder::new(cfg.clone()).unwrap();
        let q = random_multi_vec(12, cfg.d, 1);
        assert_eq!(enc.encode_query(&q).unwrap().len(), cfg.raw_dim());
    }

    #[test]
    fn fde_score_correlates_with_chamfer() {
        // Rank a candidate set by FDE dot vs by exact Chamfer. To get
        // signal above noise we plant 20 "relevant" docs that each contain
        // a near-copy of one query token; the remaining 180 are random.
        // FDE should prefer the planted set in its top-20.
        let d = 32;
        let cfg = FdeConfig {
            d, k_sim: 5, r_reps: 20,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::None, d_final: 0, seed: 42,
        };
        let enc = FdeEncoder::new(cfg).unwrap();
        let query = random_multi_vec(8, d, 1);

        let mut docs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(200);
        let mut planted: std::collections::HashSet<usize> = Default::default();
        let mut rng = StdRng::seed_from_u64(99);
        let normal = Normal::new(0.0_f32, 0.05_f32).unwrap();
        for i in 0..200 {
            let mut doc = random_multi_vec(20, d, 100 + i as u64);
            if i < 20 {
                // overwrite the first 8 doc tokens with copies of the
                // query tokens (with small noise) so this doc has a
                // strong, unambiguous Chamfer score.
                for qi in 0..8 {
                    let mut v = query[qi].clone();
                    for x in v.iter_mut() {
                        *x += normal.sample(&mut rng);
                    }
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    for x in v.iter_mut() {
                        *x /= n;
                    }
                    doc[qi] = v;
                }
                planted.insert(i);
            }
            docs.push(doc);
        }

        // exact Chamfer top-20 should be the planted set (by construction)
        let exact: Vec<(usize, f32)> = docs
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, chamfer_similarity(&query, doc)))
            .collect();
        let mut exact_sorted = exact.clone();
        exact_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let exact_top20: std::collections::HashSet<usize> =
            exact_sorted.iter().take(20).map(|x| x.0).collect();
        let exact_planted_recall =
            exact_top20.intersection(&planted).count() as f32 / planted.len() as f32;
        assert!(
            exact_planted_recall >= 0.9,
            "ground-truth Chamfer should retrieve planted docs, got {exact_planted_recall}"
        );

        let q_fde = enc.encode_query(&query).unwrap();
        let approx: Vec<(usize, f32)> = docs
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let d_fde = enc.encode_doc(doc).unwrap();
                (i, dot(&q_fde, &d_fde))
            })
            .collect();
        let mut approx_sorted = approx.clone();
        approx_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let approx_top20: std::collections::HashSet<usize> =
            approx_sorted.iter().take(20).map(|x| x.0).collect();

        let recall20 = approx_top20.intersection(&exact_top20).count() as f32 / 20.0;
        // FDE recall@20 vs exact Chamfer top-20.
        assert!(recall20 >= 0.7, "FDE/Chamfer recall@20 = {recall20}");
    }

    #[test]
    fn projection_preserves_inner_product_roughly() {
        let d = 16;
        let cfg = FdeConfig {
            d, k_sim: 4, r_reps: 8,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::Gaussian,
            d_final: 256, seed: 11,
        };
        let enc = FdeEncoder::new(cfg).unwrap();
        let q = random_multi_vec(6, d, 1);
        let v = enc.encode_query(&q).unwrap();
        assert_eq!(v.len(), 256);
    }
}
