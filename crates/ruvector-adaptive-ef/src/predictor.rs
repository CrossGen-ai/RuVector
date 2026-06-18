//! Three `ef`-predictor strategies.
//!
//! All implement [`EfPredictor`] so they can be swapped at the call site.

use crate::features::QueryFeatures;

/// Maps online query features to a recommended `ef` value.
pub trait EfPredictor: Send + Sync {
    fn predict(&self, features: &QueryFeatures) -> usize;
    fn name(&self) -> &'static str;
}

/// **Baseline**: classical fixed-ef HNSW search.
#[derive(Debug, Clone, Copy)]
pub struct FixedEf {
    pub ef: usize,
}
impl FixedEf {
    pub fn new(ef: usize) -> Self {
        Self { ef }
    }
}
impl EfPredictor for FixedEf {
    fn predict(&self, _f: &QueryFeatures) -> usize {
        self.ef
    }
    fn name(&self) -> &'static str {
        "FixedEf"
    }
}

/// **Heuristic adaptive ef** (LAET-inspired).
///
/// Idea: queries with a small `min_over_mean` ratio are "easy" (one
/// pivot is clearly closer than the others ⇒ a well-defined cluster
/// home).  Queries near boundaries (`min_over_mean` close to 1.0) need
/// more exploration.  Out-of-distribution queries (`min_pivot_d2`
/// large in absolute terms) also need more `ef`.
///
/// The mapping is intentionally simple and explainable — no training
/// data needed.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicAdaptiveEf {
    pub ef_min: usize,
    pub ef_max: usize,
    /// OOD threshold: if `min_pivot_d2 > ood_d2`, treat the query as
    /// out-of-distribution and push to `ef_max`.
    pub ood_d2: f32,
}
impl HeuristicAdaptiveEf {
    pub fn new(ef_min: usize, ef_max: usize, ood_d2: f32) -> Self {
        assert!(ef_min <= ef_max);
        Self { ef_min, ef_max, ood_d2 }
    }
}
impl EfPredictor for HeuristicAdaptiveEf {
    fn predict(&self, f: &QueryFeatures) -> usize {
        if f.min_pivot_d2 > self.ood_d2 {
            return self.ef_max;
        }
        // `min_over_mean` is in [0, 1].  Map linearly: 0 ⇒ ef_min, 1 ⇒ ef_max.
        let r = f.min_over_mean.clamp(0.0, 1.0);
        let span = (self.ef_max - self.ef_min) as f32;
        let ef = self.ef_min as f32 + r * span;
        ef.round() as usize
    }
    fn name(&self) -> &'static str {
        "HeuristicAdaptiveEf"
    }
}

/// One training sample: `(features, min_ef_observed_to_hit_target_recall)`.
#[derive(Debug, Clone, Copy)]
pub struct TrainingSample {
    pub features: QueryFeatures,
    pub min_ef_for_target: f32,
}

/// **Learned adaptive ef** (Auncel-style ridge regression).
///
/// We collect a labelled training set offline by running every query at
/// a sweep of `ef` values and recording the *smallest* `ef` for which
/// the query hit the target recall.  Then we fit a tiny linear model:
///
/// ```text
///   ef_predicted = max(ef_floor, round( w · features ))
/// ```
///
/// The fit is closed-form ridge regression — small enough to be useful
/// at edge / embedded scales (the model is 5 floats).
#[derive(Debug, Clone)]
pub struct LearnedAdaptiveEf {
    /// `[bias, w_min, w_mean, w_std, w_min_over_mean]`.
    weights: [f32; 5],
    ef_min: usize,
    ef_max: usize,
}

impl LearnedAdaptiveEf {
    pub fn weights(&self) -> [f32; 5] {
        self.weights
    }

    /// Fit via closed-form ridge regression.
    ///
    /// Solves `(XᵀX + λI) w = Xᵀy` using a 5×5 explicit inverse (small,
    /// so numerical stability of plain inversion is fine for our scale).
    ///
    /// # Panics
    /// Panics if `samples` is empty.
    pub fn fit(samples: &[TrainingSample], lambda: f32, ef_min: usize, ef_max: usize) -> Self {
        assert!(!samples.is_empty());
        const D: usize = 5;
        let mut xtx = [[0.0f64; D]; D];
        let mut xty = [0.0f64; D];
        for s in samples {
            let x = s.features.to_input();
            let y = s.min_ef_for_target as f64;
            for i in 0..D {
                for j in 0..D {
                    xtx[i][j] += x[i] as f64 * x[j] as f64;
                }
                xty[i] += x[i] as f64 * y;
            }
        }
        // Ridge: add λ to diagonal (skip bias term — common practice).
        for i in 1..D {
            xtx[i][i] += lambda as f64;
        }
        let inv = invert_5x5(&xtx).expect("XᵀX + λI must be invertible");
        let mut w = [0.0f32; D];
        for i in 0..D {
            let mut s = 0.0f64;
            for j in 0..D {
                s += inv[i][j] * xty[j];
            }
            w[i] = s as f32;
        }
        Self { weights: w, ef_min, ef_max }
    }
}

impl EfPredictor for LearnedAdaptiveEf {
    fn predict(&self, f: &QueryFeatures) -> usize {
        let x = f.to_input();
        let mut s = 0.0f32;
        for i in 0..5 {
            s += self.weights[i] * x[i];
        }
        let ef = s.round() as i64;
        ef.max(self.ef_min as i64).min(self.ef_max as i64) as usize
    }
    fn name(&self) -> &'static str {
        "LearnedAdaptiveEf"
    }
}

/// Gauss–Jordan inversion of a small dense matrix.  Returns `None` if
/// the matrix is singular.
fn invert_5x5(m: &[[f64; 5]; 5]) -> Option<[[f64; 5]; 5]> {
    const D: usize = 5;
    let mut a = [[0.0f64; 10]; D];
    for i in 0..D {
        for j in 0..D {
            a[i][j] = m[i][j];
        }
        a[i][D + i] = 1.0;
    }
    for i in 0..D {
        // Pivot.
        let mut pivot = i;
        for r in (i + 1)..D {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        if pivot != i {
            a.swap(i, pivot);
        }
        let p = a[i][i];
        for j in 0..(2 * D) {
            a[i][j] /= p;
        }
        for r in 0..D {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for j in 0..(2 * D) {
                a[r][j] -= f * a[i][j];
            }
        }
    }
    let mut out = [[0.0f64; 5]; 5];
    for i in 0..D {
        for j in 0..D {
            out[i][j] = a[i][D + j];
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qf(min: f32, mean: f32, std: f32) -> QueryFeatures {
        QueryFeatures {
            min_pivot_d2: min,
            mean_pivot_d2: mean,
            std_pivot_d2: std,
            min_over_mean: if mean > 0.0 { min / mean } else { 0.0 },
        }
    }

    #[test]
    fn fixed_predictor_returns_constant() {
        let p = FixedEf::new(64);
        assert_eq!(p.predict(&qf(0.1, 1.0, 0.5)), 64);
        assert_eq!(p.predict(&qf(100.0, 1.0, 0.5)), 64);
    }

    #[test]
    fn heuristic_pushes_ood_to_max() {
        let p = HeuristicAdaptiveEf::new(16, 256, 50.0);
        assert_eq!(p.predict(&qf(1000.0, 2.0, 1.0)), 256);
    }

    #[test]
    fn heuristic_interpolates_in_id_range() {
        let p = HeuristicAdaptiveEf::new(16, 256, 50.0);
        let easy = qf(0.1, 1.0, 0.5); // ratio 0.1 ⇒ near ef_min
        let hard = qf(0.95, 1.0, 0.5); // ratio 0.95 ⇒ near ef_max
        let ef_easy = p.predict(&easy);
        let ef_hard = p.predict(&hard);
        assert!(ef_easy < ef_hard);
        assert!(ef_easy >= 16 && ef_hard <= 256);
    }

    #[test]
    fn ridge_fit_recovers_linear_function() {
        // Synthetic: true model is ef = 8 + 5 * min_over_mean (pure feature 4).
        let samples: Vec<TrainingSample> = (0..50)
            .map(|i| {
                let r = i as f32 / 50.0;
                let f = qf(r, 1.0, 0.5);
                TrainingSample {
                    features: f,
                    min_ef_for_target: 8.0 + 5.0 * r,
                }
            })
            .collect();
        let model = LearnedAdaptiveEf::fit(&samples, 1e-3, 1, 1024);
        // Check predictions are close on a held-out point.
        let test = qf(0.7, 1.0, 0.5);
        let pred = model.predict(&test);
        let truth = (8.0_f32 + 5.0 * 0.7).round() as usize;
        assert!(pred.abs_diff(truth) <= 1, "pred={pred}, truth={truth}");
    }
}
