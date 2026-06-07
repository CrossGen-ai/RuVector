//! Synthetic content-drift simulator.
//!
//! We model the data distribution as a Gaussian mixture with K modes whose
//! relative weights shift over time. Two knobs control "how drifted" a
//! generated batch is:
//!
//!   * `mode_translation` — at time t the mean of mode k moves by
//!     `mode_translation * t * mode_drift_dir[k]`. Captures *concept drift*
//!     (the same cluster shifts in embedding space).
//!   * `weight_rotation`  — at time t the mode weights are rotated through a
//!     softmax over `base_weights + t * direction`. Captures *prevalence
//!     drift* (some clusters become more popular).
//!
//! The result is a deterministic, reproducible drift profile that produces
//! materially different recall numbers between a stale IVF index and one that
//! is incrementally rebalanced.

use crate::SmallRng;

#[derive(Clone)]
pub struct DriftWorld {
    pub dim: usize,
    pub n_modes: usize,
    pub seed: u64,
    pub mode_means: Vec<Vec<f32>>,
    pub mode_drift_dir: Vec<Vec<f32>>,
    pub mode_sigma: f32,
    pub mode_translation: f32,
    pub base_weights: Vec<f32>,
    pub weight_drift_dir: Vec<f32>,
    pub weight_rotation: f32,
}

impl DriftWorld {
    pub fn new(dim: usize, n_modes: usize, seed: u64) -> Self {
        let mut rng = SmallRng::new(seed);
        let mode_means: Vec<Vec<f32>> = (0..n_modes)
            .map(|_| (0..dim).map(|_| 2.5 * rng.normal()).collect())
            .collect();
        let mode_drift_dir: Vec<Vec<f32>> = (0..n_modes)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
                let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect();
        let base_weights = vec![1.0 / n_modes as f32; n_modes];
        let mut weight_drift_dir: Vec<f32> = (0..n_modes).map(|_| rng.normal()).collect();
        let m = weight_drift_dir.iter().map(|x| x.abs()).sum::<f32>().max(1.0);
        for x in &mut weight_drift_dir {
            *x /= m;
        }
        Self {
            dim,
            n_modes,
            seed,
            mode_means,
            mode_drift_dir,
            mode_sigma: 0.6,
            mode_translation: 0.15,
            base_weights,
            weight_drift_dir,
            weight_rotation: 0.6,
        }
    }

    /// Mode weights at time `t` (softmax(base + t * dir)).
    pub fn weights_at(&self, t: f32) -> Vec<f32> {
        let mut logits: Vec<f32> = self
            .base_weights
            .iter()
            .zip(self.weight_drift_dir.iter())
            .map(|(b, d)| b + self.weight_rotation * t * d)
            .collect();
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for x in &mut logits {
            *x = (*x - m).exp();
        }
        let s = logits.iter().sum::<f32>().max(1e-6);
        for x in &mut logits {
            *x /= s;
        }
        logits
    }

    /// Mode mean at time `t` (base + translation * t * direction).
    pub fn mean_at(&self, mode: usize, t: f32) -> Vec<f32> {
        let mut out = self.mode_means[mode].clone();
        for d in 0..self.dim {
            out[d] += self.mode_translation * t * self.mode_drift_dir[mode][d];
        }
        out
    }

    /// Draw `n` vectors at time `t` with the current mixture.
    pub fn batch(&self, n: usize, t: f32, rng: &mut SmallRng) -> Vec<Vec<f32>> {
        let weights = self.weights_at(t);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mode = sample_categorical(&weights, rng);
            let mu = self.mean_at(mode, t);
            let mut v = Vec::with_capacity(self.dim);
            for d in 0..self.dim {
                v.push(mu[d] + self.mode_sigma * rng.normal());
            }
            out.push(v);
        }
        out
    }
}

fn sample_categorical(w: &[f32], rng: &mut SmallRng) -> usize {
    let u = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in w.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    w.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let w = DriftWorld::new(4, 5, 7);
        let p = w.weights_at(2.5);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for x in p {
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn mean_moves_with_time() {
        let w = DriftWorld::new(4, 3, 7);
        let m0 = w.mean_at(0, 0.0);
        let m5 = w.mean_at(0, 5.0);
        let d: f32 = m0
            .iter()
            .zip(m5.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(d > 1e-3, "mean must drift");
    }

    #[test]
    fn batch_is_reproducible() {
        let w = DriftWorld::new(4, 3, 9);
        let a = w.batch(8, 1.5, &mut SmallRng::new(11));
        let b = w.batch(8, 1.5, &mut SmallRng::new(11));
        assert_eq!(a, b);
    }
}
