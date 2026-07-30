//! Closed-form ridge regression trainer. No ML deps.
//!
//! Given labelled (features, target) rows, solve (XᵀX + λI) w = Xᵀ y via Gauss-Jordan.
//! We predict a stopping score in [0, 1]: higher => keep searching.
//! Training target is `remaining_improvement_potential` (see `train.rs::build_dataset`).

use crate::features::Features;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearModel {
    /// Weights over `[bias, iter, best_dist, delta_best_dist, iters_since_improve, ef_min_reached]`.
    pub w: Vec<f32>,
    /// Predictor threshold: if score(x) < threshold, stop.
    pub threshold: f32,
}

impl LinearModel {
    #[inline]
    pub fn score(&self, f: &Features) -> f32 {
        let a = f.as_array();
        let mut s = self.w[0];
        for i in 0..Features::DIM {
            s += self.w[i + 1] * a[i];
        }
        s
    }
}

/// Fit ridge regression by Gauss-Jordan elimination. `x_rows` are Features::DIM long.
pub fn fit_ridge(x_rows: &[[f32; Features::DIM]], y: &[f32], lambda: f32) -> Vec<f32> {
    let d = Features::DIM + 1; // + bias
    let n = x_rows.len();
    assert_eq!(n, y.len());

    // Build augmented matrix A = XᵀX + λI ; b = Xᵀ y (excluding bias reg).
    let mut a = vec![0f64; d * d];
    let mut b = vec![0f64; d];
    for row in 0..n {
        let mut xi = [1f64; Features::DIM + 1];
        for k in 0..Features::DIM {
            xi[k + 1] = x_rows[row][k] as f64;
        }
        let yi = y[row] as f64;
        for i in 0..d {
            b[i] += xi[i] * yi;
            for j in 0..d {
                a[i * d + j] += xi[i] * xi[j];
            }
        }
    }
    for i in 1..d {
        a[i * d + i] += lambda as f64; // ridge (skip bias)
    }

    // Solve Aw = b via Gauss-Jordan.
    let mut m = vec![0f64; d * (d + 1)];
    for i in 0..d {
        for j in 0..d {
            m[i * (d + 1) + j] = a[i * d + j];
        }
        m[i * (d + 1) + d] = b[i];
    }
    for col in 0..d {
        // Pivot: largest abs in column at rows >= col.
        let mut piv = col;
        let mut best = m[col * (d + 1) + col].abs();
        for r in (col + 1)..d {
            let v = m[r * (d + 1) + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            // Singular — return zeros of appropriate shape.
            return vec![0f32; d];
        }
        if piv != col {
            for j in 0..=d {
                m.swap(col * (d + 1) + j, piv * (d + 1) + j);
            }
        }
        let inv = 1.0 / m[col * (d + 1) + col];
        for j in 0..=d {
            m[col * (d + 1) + j] *= inv;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = m[r * (d + 1) + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..=d {
                m[r * (d + 1) + j] -= f * m[col * (d + 1) + j];
            }
        }
    }
    (0..d).map(|i| m[i * (d + 1) + d] as f32).collect()
}
