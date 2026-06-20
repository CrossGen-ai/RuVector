//! Tiny ridge-regression predictor mapping per-query features to the
//! minimum `ef` that achieves a target recall.
//!
//! We solve the normal equations directly via Gauss-Jordan elimination
//! over `f64` — fast and dependency-free for the small feature counts
//! we use (≤16). For production scale you'd replace this with a
//! gradient-boosted regressor (e.g. LightGBM via the `lightgbm` crate),
//! but ridge already captures most of the available signal.

/// Per-query features fed into the LAET predictor. All values are
/// derived from the cheap upper-layer descent (which every HNSW query
/// already runs), so feature extraction is essentially free.
#[derive(Clone, Debug, Default)]
pub struct LaetFeatures {
    /// L2 norm of the query (informative when the dataset clusters
    /// near the origin).
    pub q_norm: f32,
    /// Distance to the layer-0 entry point produced by upper-layer
    /// descent. Smaller ⇒ query is "deep inside" a dense region ⇒
    /// likely needs fewer probes.
    pub d_entry: f32,
    /// Number of distance computations spent on the upper-layer
    /// descent. A long descent already hints at a hard query.
    pub descent_dists: f32,
    /// Final-vs-initial best-distance ratio on the upper layers.
    /// Small ratio ⇒ descent made big progress ⇒ usually easy.
    pub descent_ratio: f32,
}

impl LaetFeatures {
    pub fn to_vec(&self) -> [f64; 5] {
        // Include bias term.
        [
            1.0,
            self.q_norm as f64,
            self.d_entry as f64,
            self.descent_dists as f64,
            self.descent_ratio as f64,
        ]
    }
}

#[derive(Clone, Debug)]
pub struct RidgePredictor {
    pub weights: [f64; 5],
    pub lambda: f64,
    pub ef_floor: usize,
    pub ef_ceil: usize,
}

impl Default for RidgePredictor {
    fn default() -> Self {
        Self {
            weights: [0.0; 5],
            lambda: 1.0,
            ef_floor: 8,
            ef_ceil: 256,
        }
    }
}

impl RidgePredictor {
    /// Fit (X^T X + λI)^-1 X^T y in place. `xs[i]` is a row of length
    /// 5 (already with bias), `ys[i]` is the target `ef`.
    pub fn fit(&mut self, xs: &[[f64; 5]], ys: &[f64]) {
        assert_eq!(xs.len(), ys.len());
        let d = 5;
        // Build A = X^T X + λI and b = X^T y
        let mut a = [[0.0f64; 5]; 5];
        let mut b = [0.0f64; 5];
        for (row, &y) in xs.iter().zip(ys.iter()) {
            for i in 0..d {
                b[i] += row[i] * y;
                for j in 0..d {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
        for i in 0..d {
            a[i][i] += self.lambda;
        }
        // Augment and solve via Gauss-Jordan.
        let mut m = [[0.0f64; 6]; 5];
        for i in 0..d {
            for j in 0..d {
                m[i][j] = a[i][j];
            }
            m[i][d] = b[i];
        }
        for k in 0..d {
            // Partial pivot
            let mut piv = k;
            for r in k + 1..d {
                if m[r][k].abs() > m[piv][k].abs() {
                    piv = r;
                }
            }
            if piv != k {
                m.swap(k, piv);
            }
            let p = m[k][k];
            if p.abs() < 1e-12 {
                // singular — fall back to zero weights for this column.
                continue;
            }
            for j in k..=d {
                m[k][j] /= p;
            }
            for r in 0..d {
                if r == k {
                    continue;
                }
                let f = m[r][k];
                if f == 0.0 {
                    continue;
                }
                for j in k..=d {
                    m[r][j] -= f * m[k][j];
                }
            }
        }
        for i in 0..d {
            self.weights[i] = m[i][d];
        }
    }

    pub fn predict(&self, f: &LaetFeatures) -> usize {
        let x = f.to_vec();
        let mut s = 0.0;
        for i in 0..5 {
            s += self.weights[i] * x[i];
        }
        let v = s.round() as i64;
        v.clamp(self.ef_floor as i64, self.ef_ceil as i64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_linear_targets() {
        // y = 10 + 2 * d_entry  exactly
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..50 {
            let f = LaetFeatures {
                q_norm: 0.0,
                d_entry: i as f32,
                descent_dists: 0.0,
                descent_ratio: 0.0,
            };
            xs.push(f.to_vec());
            ys.push(10.0 + 2.0 * i as f64);
        }
        let mut p = RidgePredictor {
            lambda: 1e-6,
            ..Default::default()
        };
        p.fit(&xs, &ys);
        let pred = p.predict(&LaetFeatures {
            q_norm: 0.0,
            d_entry: 7.0,
            descent_dists: 0.0,
            descent_ratio: 0.0,
        });
        // 10 + 2*7 = 24
        assert!((pred as i32 - 24).abs() <= 1, "got {pred}");
    }
}
