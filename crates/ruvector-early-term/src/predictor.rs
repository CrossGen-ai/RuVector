//! Ridge regression in closed form: w = (XᵀX + λI)⁻¹ Xᵀy.
//!
//! Features per step (8-dim, all finite, in [0,1] range after the
//! normalizations below):
//!   f0: log10(step + 1) / 4               (step counter, capped)
//!   f1: step / ef_max                     (progress through budget)
//!   f2: kth_dist                          (current k-th best, ~[0, 2])
//!   f3: cand_dist - kth_dist              (slack of current candidate)
//!   f4: kth_history mean over window
//!   f5: kth_history stddev over window
//!   f6: (history[-1] - history[-w]) / w   (slope)
//!   f7: 1                                  (bias)
//!
//! Target: residual recall risk = max(0, target_recall - current_recall_at_ef_max).
//! We train so that "predict low → safe to stop".

pub struct QueryFeatures(pub [f32; 8]);

impl QueryFeatures {
    pub fn from_history(
        step: u32,
        cand_dist: f32,
        kth_dist: f32,
        history: &[f32],
        _cand_history: &[f32],
        ef_max: usize,
    ) -> Self {
        let w = 8usize.min(history.len());
        let window = &history[history.len().saturating_sub(w)..];
        let mean = window.iter().copied().sum::<f32>() / w.max(1) as f32;
        let var = window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / w.max(1) as f32;
        let slope = if window.len() >= 2 {
            (window[window.len() - 1] - window[0]) / (window.len() as f32 - 1.0)
        } else {
            0.0
        };
        let f = [
            ((step as f32 + 1.0).log10() / 4.0).min(1.0),
            (step as f32 / ef_max.max(1) as f32).min(2.0),
            kth_dist.clamp(0.0, 2.0),
            (cand_dist - kth_dist).clamp(-2.0, 2.0),
            mean.clamp(0.0, 2.0),
            var.sqrt().clamp(0.0, 1.0),
            slope.clamp(-1.0, 1.0),
            1.0,
        ];
        QueryFeatures(f)
    }
    pub fn to_vec(&self) -> Vec<f32> { self.0.to_vec() }
}

#[derive(Clone)]
pub struct RidgeRegressor {
    pub weights: Vec<f32>,
}

impl RidgeRegressor {
    pub fn new(dim: usize) -> Self {
        Self { weights: vec![0.0; dim] }
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        let mut s = 0.0f32;
        let d = self.weights.len().min(x.len());
        for i in 0..d { s += self.weights[i] * x[i]; }
        s
    }

    /// Train via closed-form ridge regression. X is `n x d` (row-major),
    /// y is `n`. lambda is the L2 penalty (>0 to ensure invertibility).
    pub fn train(x: &[Vec<f32>], y: &[f32], lambda: f32) -> Self {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        if n == 0 { return Self::new(0); }
        let d = x[0].len();
        // A = XᵀX + λI  (d x d)
        let mut a = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0f32;
                for r in 0..n { s += x[r][i] * x[r][j]; }
                a[i * d + j] = s;
                if i == j { a[i * d + j] += lambda; }
            }
        }
        // b = Xᵀy  (d)
        let mut b = vec![0.0f32; d];
        for i in 0..d {
            let mut s = 0.0f32;
            for r in 0..n { s += x[r][i] * y[r]; }
            b[i] = s;
        }
        // Solve via Gaussian elimination with partial pivoting.
        let w = solve(&mut a, &mut b, d);
        Self { weights: w }
    }
}

fn solve(a: &mut [f32], b: &mut [f32], d: usize) -> Vec<f32> {
    // Augmented in-place elimination.
    for k in 0..d {
        // Pivot
        let mut piv = k;
        let mut best = a[k * d + k].abs();
        for r in (k + 1)..d {
            let v = a[r * d + k].abs();
            if v > best { best = v; piv = r; }
        }
        if piv != k {
            for c in 0..d {
                a.swap(k * d + c, piv * d + c);
            }
            b.swap(k, piv);
        }
        let pv = a[k * d + k];
        if pv.abs() < 1e-9 { continue; }
        for r in (k + 1)..d {
            let f = a[r * d + k] / pv;
            for c in k..d {
                a[r * d + c] -= f * a[k * d + c];
            }
            b[r] -= f * b[k];
        }
    }
    let mut x = vec![0.0f32; d];
    for i in (0..d).rev() {
        let mut s = b[i];
        for j in (i + 1)..d {
            s -= a[i * d + j] * x[j];
        }
        let pv = a[i * d + i];
        x[i] = if pv.abs() < 1e-9 { 0.0 } else { s / pv };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_linear_function() {
        // y = 2*x0 + -1*x1 + 0.5 (bias)
        let xs: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                let a = (i as f32 * 0.013).sin();
                let b = (i as f32 * 0.071).cos();
                vec![a, b, 1.0]
            })
            .collect();
        let ys: Vec<f32> = xs.iter().map(|r| 2.0 * r[0] - 1.0 * r[1] + 0.5 * r[2]).collect();
        let reg = RidgeRegressor::train(&xs, &ys, 1e-4);
        let pred = reg.predict(&[0.3, -0.4, 1.0]);
        let want = 2.0 * 0.3 - 1.0 * -0.4 + 0.5;
        assert!((pred - want).abs() < 0.01, "pred={} want={}", pred, want);
    }
}
