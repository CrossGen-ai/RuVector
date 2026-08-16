//! Tiny logistic-regression classifier for early-termination decisions.
//!
//! Five runtime features drive a linear model whose sigmoid output is
//! `P(top-k will still change if we keep expanding)`. The features are extracted
//! from the beam-search state after every neighbour-expansion step.
//!
//! Training uses plain SGD with logistic loss. No external ML dependencies —
//! implementation is pure Rust and dependency-free.

/// A snapshot of features extracted from one search step.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureSnapshot {
    /// Current closest distance in the results heap.
    pub best_dist: f32,
    /// Moving-average decrease of `best_dist` over the last few steps.
    pub improve_rate: f32,
    /// Normalised gap between k-th and (k-1)-th result: (d_k - d_{k-1}) / (d_k + eps).
    pub gap_kth: f32,
    /// Fraction of ef budget consumed: steps_so_far / ef.
    pub steps_norm: f32,
    /// Fraction of the currently-expanded node's neighbours that were unvisited.
    pub frontier_ratio: f32,
}

impl FeatureSnapshot {
    /// Pack into an array suitable for dot-product with weights[1..].
    pub fn as_array(&self) -> [f32; 5] {
        [
            self.best_dist,
            self.improve_rate,
            self.gap_kth,
            self.steps_norm,
            self.frontier_ratio,
        ]
    }
}

/// Logistic classifier with 5 feature weights + 1 bias.
///
/// Layout: `w[0]` = bias, `w[1..=5]` = feature weights.
#[derive(Debug, Clone)]
pub struct LogisticPredictor {
    pub w: [f32; 6],
}

impl Default for LogisticPredictor {
    fn default() -> Self {
        // Reasonable hand-set fallback that agrees with the trained model's sign
        // pattern in most runs. Retrain via `LogisticPredictor::train` for the
        // actual dataset in use.
        Self {
            w: [-1.5, 3.0, 4.0, -1.0, -2.0, 1.5],
        }
    }
}

/// Training hyperparameters for SGD.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub lr: f32,
    pub epochs: usize,
    pub l2: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.05,
            epochs: 40,
            l2: 1e-4,
            seed: 42,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LogisticPredictor {
    /// Probability that continuing expansion will improve the top-k.
    pub fn p_improve(&self, f: &FeatureSnapshot) -> f32 {
        let a = f.as_array();
        let z = self.w[0]
            + self.w[1] * a[0]
            + self.w[2] * a[1]
            + self.w[3] * a[2]
            + self.w[4] * a[3]
            + self.w[5] * a[4];
        sigmoid(z)
    }

    /// Decide: keep expanding if p_improve >= tau, otherwise stop.
    pub fn should_continue(&self, f: &FeatureSnapshot, tau: f32) -> bool {
        self.p_improve(f) >= tau
    }

    /// SGD training. `samples` = (features, label∈{0,1}). Label 1 means
    /// "top-k still changed at or after this step".
    pub fn train(samples: &[(FeatureSnapshot, f32)], cfg: &TrainConfig) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        // Standardise features (zero mean, unit variance) — massively improves
        // convergence for a mixed-scale feature set.
        let (mu, sigma) = compute_stats(samples);
        let mut w = [0.0_f32; 6];
        let mut rng_state = cfg.seed;
        let n = samples.len();

        for _ in 0..cfg.epochs {
            // Shuffle indices Fisher–Yates style with LCG rand.
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (rng_state >> 33) as usize % (i + 1);
                idx.swap(i, j);
            }
            for &i in &idx {
                let (feat, label) = &samples[i];
                let x = standardise(feat.as_array(), &mu, &sigma);
                let z = w[0]
                    + w[1] * x[0]
                    + w[2] * x[1]
                    + w[3] * x[2]
                    + w[4] * x[3]
                    + w[5] * x[4];
                let p = sigmoid(z);
                let g = p - *label;
                w[0] -= cfg.lr * g;
                for j in 0..5 {
                    w[j + 1] -= cfg.lr * (g * x[j] + cfg.l2 * w[j + 1]);
                }
            }
        }
        // Bake standardisation into weights so inference is a single dot-product
        // over raw features:  z = w0' + sum_j (w_{j+1} / sigma_j) * (x_j - mu_j)
        //                    = (w0' - sum_j (w_{j+1} mu_j / sigma_j))
        //                       + sum_j (w_{j+1} / sigma_j) * x_j
        let mut baked = [0.0_f32; 6];
        baked[0] = w[0];
        for j in 0..5 {
            let s = sigma[j].max(1e-6);
            let wj = w[j + 1] / s;
            baked[0] -= wj * mu[j];
            baked[j + 1] = wj;
        }
        Self { w: baked }
    }
}

fn compute_stats(samples: &[(FeatureSnapshot, f32)]) -> ([f32; 5], [f32; 5]) {
    let n = samples.len() as f32;
    let mut mu = [0.0_f32; 5];
    for (f, _) in samples {
        let a = f.as_array();
        for j in 0..5 {
            mu[j] += a[j];
        }
    }
    for j in 0..5 {
        mu[j] /= n;
    }
    let mut var = [0.0_f32; 5];
    for (f, _) in samples {
        let a = f.as_array();
        for j in 0..5 {
            let d = a[j] - mu[j];
            var[j] += d * d;
        }
    }
    let mut sigma = [0.0_f32; 5];
    for j in 0..5 {
        sigma[j] = (var[j] / n).sqrt().max(1e-6);
    }
    (mu, sigma)
}

fn standardise(x: [f32; 5], mu: &[f32; 5], sigma: &[f32; 5]) -> [f32; 5] {
    let mut o = [0.0_f32; 5];
    for j in 0..5 {
        o[j] = (x[j] - mu[j]) / sigma[j].max(1e-6);
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!(sigmoid(-1000.0) < 1e-6);
        assert!(sigmoid(1000.0) > 1.0 - 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_predictor_stops_on_low_improve() {
        let p = LogisticPredictor::default();
        // Low improve_rate + high steps_norm + tiny gap → should stop.
        let f = FeatureSnapshot {
            best_dist: 0.001,
            improve_rate: 0.0,
            gap_kth: 0.001,
            steps_norm: 1.0,
            frontier_ratio: 0.0,
        };
        assert!(p.p_improve(&f) < 0.5);
    }

    #[test]
    fn training_separates_two_classes() {
        // Synthetic: label=1 when improve_rate>0.5, label=0 otherwise.
        let mut samples = Vec::new();
        for i in 0..40 {
            let r = if i % 2 == 0 { 1.0 } else { 0.0 };
            let f = FeatureSnapshot {
                best_dist: 0.5,
                improve_rate: if r > 0.5 { 0.8 } else { 0.05 },
                gap_kth: 0.01,
                steps_norm: 0.5,
                frontier_ratio: 0.5,
            };
            samples.push((f, r));
        }
        let p = LogisticPredictor::train(
            &samples,
            &TrainConfig {
                lr: 0.1,
                epochs: 80,
                ..TrainConfig::default()
            },
        );
        let pos = FeatureSnapshot {
            best_dist: 0.5,
            improve_rate: 0.8,
            gap_kth: 0.01,
            steps_norm: 0.5,
            frontier_ratio: 0.5,
        };
        let neg = FeatureSnapshot {
            best_dist: 0.5,
            improve_rate: 0.05,
            gap_kth: 0.01,
            steps_norm: 0.5,
            frontier_ratio: 0.5,
        };
        let pp = p.p_improve(&pos);
        let pn = p.p_improve(&neg);
        assert!(
            pp - pn > 0.3,
            "trained model should separate classes: pos={pp:.3} neg={pn:.3}"
        );
    }
}
