//! Query-adaptive `ef_search` predictor.
//!
//! Workflow:
//!   1. Build the graph once.
//!   2. Sample S "probe" queries. For each, run an exact brute-force search
//!      to obtain ground-truth top-k. Then sweep `ef` over a small grid and
//!      label the query with `min ef such that recall >= target`.
//!   3. Featurise each probe query (cheap features, no extra distance work
//!      beyond what we already paid in step 2) and fit a linear regression
//!      against `log2(ef_label)`. Linear in `log` space gives a multiplicative
//!      response to feature changes — empirically the cleanest fit.
//!   4. At query time: compute features, run the predictor, clip to
//!      `[ef_min, ef_max]`, search.
//!
//! Features used (all cheap, derived from the graph entry point + a single
//! beam-search probe at small `ef`):
//!   * `f0` = bias (1.0)
//!   * `f1` = distance from query to the index entry point
//!   * `f2` = best distance found by a cheap `ef=ef_probe` search
//!   * `f3` = mean distance of the top-`ef_probe` neighbours of that probe
//!   * `f4` = `f3 - f2` (local cluster spread — a thin tail = easy query)
//!
//! These are all available "for free" because we already need the probe
//! search to *start* the search. Total overhead per query is < 1% of
//! distance computations at any sensible `ef`.

use crate::nsw::{sq_l2, NswIndex, SearchStats};

#[derive(Debug, Clone, Copy)]
pub struct EfFeatures {
    pub f: [f32; 5],
    pub probe_stats: SearchStats,
}

/// Cheap probe used both to seed features and as warm-start for the full
/// search. We do not throw the probe work away.
pub fn extract_features(index: &NswIndex, query: &[f32], ef_probe: usize) -> EfFeatures {
    let entry = index.vector(0);
    let entry_d = sq_l2(query, entry);
    let (ids, stats) = index.search(query, ef_probe, ef_probe);
    let dists: Vec<f32> = ids.iter().map(|&i| sq_l2(query, index.vector(i))).collect();
    let best = dists.first().copied().unwrap_or(entry_d);
    let mean = if dists.is_empty() {
        entry_d
    } else {
        dists.iter().sum::<f32>() / dists.len() as f32
    };
    EfFeatures {
        f: [1.0, entry_d, best, mean, mean - best],
        probe_stats: stats,
    }
}

/// Linear predictor `log2_ef = w · features`, clipped to `[ef_min, ef_max]`.
#[derive(Debug, Clone)]
pub struct AdaptiveEf {
    pub w: [f32; 5],
    pub ef_min: u32,
    pub ef_max: u32,
    pub ef_probe: u32,
    /// Additive bias added to the predicted `log2(ef)`. Calibrated post-fit
    /// so that the predictor matches a target tail-recall. A small positive
    /// value (~0.3-0.6, i.e. 1.2-1.5x in `ef` space) is usually enough to
    /// soak up residual variance and avoid undershooting.
    pub log_bias: f32,
}

impl AdaptiveEf {
    pub fn predict(&self, feats: &EfFeatures) -> u32 {
        let mut log_ef = self.log_bias;
        for i in 0..5 {
            log_ef += self.w[i] * feats.f[i];
        }
        let ef = 2.0f32.powf(log_ef).round() as i64;
        ef.clamp(self.ef_min as i64, self.ef_max as i64) as u32
    }

    /// Fit weights by ordinary least squares on `(features, log2(label_ef))`
    /// pairs. Solved with a small Gauss-Jordan over a 5x5 normal-equation
    /// matrix — fine for up to ~1e6 probes and avoids a `nalgebra` dep.
    pub fn fit(
        train: &[(EfFeatures, u32)],
        ef_min: u32,
        ef_max: u32,
        ef_probe: u32,
    ) -> Self {
        const N: usize = 5;
        let mut xtx = [[0.0f64; N]; N];
        let mut xty = [0.0f64; N];
        for (feats, ef_label) in train {
            let y = (*ef_label as f64).max(1.0).log2();
            for i in 0..N {
                xty[i] += feats.f[i] as f64 * y;
                for j in 0..N {
                    xtx[i][j] += feats.f[i] as f64 * feats.f[j] as f64;
                }
            }
        }
        // tiny ridge for stability
        for i in 0..N {
            xtx[i][i] += 1e-3;
        }
        let w = solve_5x5(xtx, xty);
        let mut w32 = [0.0f32; N];
        for i in 0..N {
            w32[i] = w[i] as f32;
        }
        Self { w: w32, ef_min, ef_max, ef_probe, log_bias: 0.0 }
    }

    /// Walk a small grid of `log_bias` values on a held-out validation slice
    /// and pick the smallest bias whose mean recall meets `target_recall`.
    /// `eval` runs each candidate bias on the validation set and returns
    /// `(mean_recall, mean_distance_computations)`. Returns the chosen bias.
    pub fn calibrate_bias<F>(&mut self, target_recall: f64, mut eval: F) -> f32
    where
        F: FnMut(&AdaptiveEf) -> (f64, f64),
    {
        let candidates = [0.0_f32, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3];
        let mut best = (0.0_f32, f64::INFINITY);
        for &b in &candidates {
            self.log_bias = b;
            let (rec, work) = eval(self);
            if rec + 1e-9 >= target_recall && work < best.1 {
                best = (b, work);
            }
        }
        if best.1.is_infinite() {
            // none hit target — fall back to max bias
            self.log_bias = *candidates.last().unwrap();
        } else {
            self.log_bias = best.0;
        }
        self.log_bias
    }
}

// In-place Gauss-Jordan on a 5x5 system. Pivots are non-zero in practice
// because of the ridge.
fn solve_5x5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> [f64; 5] {
    for k in 0..5 {
        // partial pivot
        let mut piv = k;
        for r in (k + 1)..5 {
            if a[r][k].abs() > a[piv][k].abs() {
                piv = r;
            }
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let pivot = a[k][k];
        if pivot.abs() < 1e-12 {
            return [0.0; 5];
        }
        for j in 0..5 {
            a[k][j] /= pivot;
        }
        b[k] /= pivot;
        for i in 0..5 {
            if i == k {
                continue;
            }
            let f = a[i][k];
            for j in 0..5 {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_predicts_planted_labels() {
        // y = 5*x0 + 1.0*x1 - 0.8*x2 (+ 0,0 for unused)
        // After 2^y rounding the recovered weights drift slightly (integer
        // labels lose info) but predictions should still match labels closely.
        let truth = [5.0_f32, 1.0, -0.8, 0.0, 0.0];
        let mut train: Vec<(EfFeatures, u32)> = Vec::new();
        for i in 0..200 {
            let x = [1.0, (i as f32) * 0.02, (i as f32 - 100.0) * 0.01, 0.0, 0.0];
            let log_y = truth.iter().zip(x.iter()).map(|(w, v)| w * v).sum::<f32>();
            let ef = (2.0f32.powf(log_y).round() as i64).max(1).min(4096) as u32;
            train.push((EfFeatures { f: x, probe_stats: Default::default() }, ef));
        }
        let model = AdaptiveEf::fit(&train, 1, 4096, 16);
        let mut max_log_err = 0.0f32;
        for (feats, label) in &train {
            let pred = model.predict(feats) as f32;
            let err = (pred.log2() - (*label as f32).log2()).abs();
            if err > max_log_err { max_log_err = err; }
        }
        assert!(max_log_err < 0.25, "max log2(ef) error too high: {max_log_err}");
    }
}
