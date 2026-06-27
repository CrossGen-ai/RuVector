//! Closed-form ridge-regression budget predictor.
//!
//! Given a labelled training set `(features_i, ef_oracle_i)` we fit a linear
//! model `ef = clip(round(w·x), ef_min, ef_max)` by solving the normal
//! equations `(XᵀX + λI) w = Xᵀ y` via Gauss-Jordan inversion. This is
//! deliberately simple and tiny — eight features, one matrix inversion at
//! training time, one dot product at inference time.
//!
//! The oracle for the training labels is built by binary search: for each
//! training query, find the smallest power-of-two `ef` that achieves the
//! target per-query recall. The same oracle is also used by the benchmark as
//! the lower-bound reference.

use crate::features::{extract_features, Medoids, QueryFeatures, FEATURE_DIM};
use crate::hnsw::{brute_knn, Hnsw, SearchStats};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    pub target_recall: f32,
    pub k: usize,
    pub ef_min: usize,
    pub ef_max: usize,
    pub ridge_lambda: f32,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            target_recall: 0.95,
            k: 10,
            ef_min: 8,
            ef_max: 512,
            ridge_lambda: 1e-2,
        }
    }
}

/// Oracle: smallest `ef` in `candidates` reaching `target_recall` on this query.
pub struct OracleBudget;

impl OracleBudget {
    pub fn label(
        index: &Hnsw,
        data: &[f32],
        dim: usize,
        q: &[f32],
        cfg: &PredictorConfig,
        candidates: &[usize],
    ) -> usize {
        // Smallest ef that hits target_recall, OR — for queries whose recall
        // ceiling sits below the target — the smallest ef whose recall is
        // within 1% of the maximum achievable recall on the ladder. The
        // "saturation" branch prevents the oracle from blindly returning the
        // largest ef on queries that are genuinely unreachable, which would
        // train the predictor to over-spend on hard queries with no gain.
        let gt: HashSet<u32> = brute_knn(data, dim, q, cfg.k).into_iter().map(|x| x.0).collect();
        let mut recalls: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
        for &ef in candidates {
            let (res, _) = index.search(q, cfg.k, ef);
            let hit = res.iter().filter(|(i, _)| gt.contains(i)).count() as f32;
            let recall = hit / (cfg.k as f32);
            recalls.push((ef, recall));
            if recall + 1e-6 >= cfg.target_recall {
                return ef;
            }
        }
        let max_r = recalls.iter().map(|(_, r)| *r).fold(0.0_f32, f32::max);
        for (ef, r) in &recalls {
            if *r + 0.01 >= max_r {
                return *ef;
            }
        }
        *candidates.last().unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPredictor {
    pub w: [f32; FEATURE_DIM],
    pub cfg: PredictorConfig,
    /// Per-feature scaling: (mean, inv_std) used to standardise inputs.
    pub feat_mean: [f32; FEATURE_DIM],
    pub feat_inv_std: [f32; FEATURE_DIM],
    /// Safety margin in log-2 space (e.g. +0.25 ≈ 19% headroom).
    pub log2_margin: f32,
}

impl BudgetPredictor {
    /// Fit predictor on the supplied query/label pairs. Labels are oracle `ef`
    /// values; we regress in log2-space so that doubling the budget moves the
    /// model linearly.
    pub fn fit(
        train_features: &[QueryFeatures],
        train_labels: &[usize],
        cfg: PredictorConfig,
        log2_margin: f32,
    ) -> Result<Self, crate::Error> {
        if train_features.is_empty() {
            return Err(crate::Error::NoTraining);
        }
        let n = train_features.len();
        // standardise features
        let mut feat_mean = [0.0_f32; FEATURE_DIM];
        let mut feat_m2 = [0.0_f32; FEATURE_DIM];
        for f in train_features {
            for d in 0..FEATURE_DIM {
                feat_mean[d] += f.x[d];
            }
        }
        for d in 0..FEATURE_DIM {
            feat_mean[d] /= n as f32;
        }
        for f in train_features {
            for d in 0..FEATURE_DIM {
                let v = f.x[d] - feat_mean[d];
                feat_m2[d] += v * v;
            }
        }
        let mut feat_inv_std = [0.0_f32; FEATURE_DIM];
        for d in 0..FEATURE_DIM {
            let std = (feat_m2[d] / n as f32).sqrt().max(1e-6);
            feat_inv_std[d] = 1.0 / std;
        }
        // keep bias as bias
        feat_mean[0] = 0.0;
        feat_inv_std[0] = 1.0;

        let mut x: Vec<f32> = Vec::with_capacity(n * FEATURE_DIM);
        for f in train_features {
            for d in 0..FEATURE_DIM {
                x.push((f.x[d] - feat_mean[d]) * feat_inv_std[d]);
            }
        }
        let y: Vec<f32> = train_labels.iter().map(|&l| (l as f32).log2()).collect();
        // Sample weights: 1 + log2(label) — hard queries (large oracle ef)
        // get up to ~10× more weight, preventing OLS from collapsing to the
        // common easy case.
        let w_samp: Vec<f32> = train_labels
            .iter()
            .map(|&l| 1.0 + (l as f32).log2())
            .collect();

        // Build XtX (FEATURE_DIM x FEATURE_DIM) and Xty (FEATURE_DIM)
        let p = FEATURE_DIM;
        let mut xtx = vec![0.0_f32; p * p];
        let mut xty = vec![0.0_f32; p];
        for row in 0..n {
            let r = &x[row * p..(row + 1) * p];
            let yi = y[row];
            let wi = w_samp[row];
            for i in 0..p {
                xty[i] += wi * r[i] * yi;
                for j in 0..p {
                    xtx[i * p + j] += wi * r[i] * r[j];
                }
            }
        }
        for i in 0..p {
            xtx[i * p + i] += cfg.ridge_lambda;
        }
        let inv = invert_matrix(&xtx, p).expect("ridge keeps matrix invertible");
        let mut w_vec = vec![0.0_f32; p];
        for i in 0..p {
            for j in 0..p {
                w_vec[i] += inv[i * p + j] * xty[j];
            }
        }
        let mut w = [0.0_f32; FEATURE_DIM];
        w.copy_from_slice(&w_vec);
        Ok(Self { w, cfg, feat_mean, feat_inv_std, log2_margin })
    }

    pub fn predict(&self, feats: &QueryFeatures) -> usize {
        let mut s = 0.0_f32;
        for d in 0..FEATURE_DIM {
            s += self.w[d] * ((feats.x[d] - self.feat_mean[d]) * self.feat_inv_std[d]);
        }
        // add safety margin in log2 space then convert
        let ef = (2.0_f32).powf(s + self.log2_margin).round() as i64;
        ef.clamp(self.cfg.ef_min as i64, self.cfg.ef_max as i64) as usize
    }

    pub fn predict_query(
        &self,
        q: &[f32],
        medoids: &Medoids,
        index: &Hnsw,
        stats: &mut SearchStats,
    ) -> usize {
        let f = extract_features(q, medoids, index, stats);
        self.predict(&f)
    }
}

fn invert_matrix(m: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut a = vec![0.0_f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = m[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        // partial pivot
        let mut piv = col;
        for r in col + 1..n {
            if a[r * 2 * n + col].abs() > a[piv * 2 * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * 2 * n + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..2 * n {
                a.swap(col * 2 * n + j, piv * 2 * n + j);
            }
        }
        let inv_p = 1.0 / a[col * 2 * n + col];
        for j in 0..2 * n {
            a[col * 2 * n + j] *= inv_p;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..2 * n {
                let v = a[col * 2 * n + j];
                a[r * 2 * n + j] -= factor * v;
            }
        }
    }
    let mut inv = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invert_identity() {
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_matrix(&id, 3).unwrap();
        for i in 0..9 {
            assert!((inv[i] - id[i]).abs() < 1e-6);
        }
    }
    #[test]
    fn invert_diag() {
        let d = vec![2.0, 0.0, 0.0, 4.0];
        let inv = invert_matrix(&d, 2).unwrap();
        assert!((inv[0] - 0.5).abs() < 1e-6);
        assert!((inv[3] - 0.25).abs() < 1e-6);
    }
}
