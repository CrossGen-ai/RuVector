//! Jain & Chlamtac 1985 P² algorithm — online quantile estimation.
//!
//! Constant memory (five markers), no buffering, O(1) update.
//! Used by `QuantileTerminator` to track the distribution of recent
//! "expansion improvement" deltas so we can decide when to stop
//! expanding the HNSW beam.

#[derive(Debug, Clone)]
pub struct P2Quantile {
    p: f64,
    n: [i64; 5],
    np: [f64; 5],
    dn: [f64; 5],
    q: [f64; 5],
    count: u64,
}

impl P2Quantile {
    /// Create a tracker for the p-quantile (0 < p < 1).
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p < 1.0);
        Self {
            p,
            n: [0; 5],
            np: [0.0; 5],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            q: [0.0; 5],
            count: 0,
        }
    }

    /// Number of samples observed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Current quantile estimate (returns 0.0 if fewer than 5 samples).
    pub fn quantile(&self) -> f64 {
        if self.count < 5 {
            return 0.0;
        }
        self.q[2]
    }

    /// Add a sample.
    pub fn add(&mut self, x: f64) {
        if self.count < 5 {
            let idx = self.count as usize;
            self.q[idx] = x;
            self.count += 1;
            if self.count == 5 {
                // Sort initial markers.
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for i in 0..5 {
                    self.n[i] = i as i64;
                }
                self.np = [
                    0.0,
                    2.0 * self.p,
                    4.0 * self.p,
                    2.0 + 2.0 * self.p,
                    4.0,
                ];
            }
            return;
        }

        // Find cell k.
        let k: usize = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        for i in (k + 1)..5 {
            self.n[i] += 1;
        }
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        // Adjust heights of middle markers.
        for i in 1..4 {
            let d = self.np[i] - self.n[i] as f64;
            let nl = self.n[i - 1];
            let nr = self.n[i + 1];
            if (d >= 1.0 && nr - self.n[i] > 1) || (d <= -1.0 && nl - self.n[i] < -1) {
                let s = d.signum();
                // P² parabolic prediction.
                let qi = self.q[i];
                let qm1 = self.q[i - 1];
                let qp1 = self.q[i + 1];
                let ni = self.n[i] as f64;
                let nl = nl as f64;
                let nr = nr as f64;
                let parabolic = qi
                    + s / (nr - nl)
                        * ((ni - nl + s) * (qp1 - qi) / (nr - ni)
                            + (nr - ni - s) * (qi - qm1) / (ni - nl));
                let chosen = if parabolic > qm1 && parabolic < qp1 {
                    parabolic
                } else {
                    // Linear fallback.
                    let s_i = s as i64;
                    let nbr = self.n[(i as i64 + s_i) as usize] as f64;
                    qi + s * (self.q[(i as i64 + s_i) as usize] - qi) / (nbr - ni)
                };
                self.q[i] = chosen;
                self.n[i] += s as i64;
            }
        }

        self.count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn p2_tracks_median_of_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut p2 = P2Quantile::new(0.5);
        for _ in 0..20_000 {
            p2.add(rng.gen::<f64>());
        }
        let est = p2.quantile();
        assert!(
            (est - 0.5).abs() < 0.02,
            "median estimate off: got {est}, expected ~0.5"
        );
    }

    #[test]
    fn p2_tracks_p90_of_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut p2 = P2Quantile::new(0.9);
        for _ in 0..20_000 {
            p2.add(rng.gen::<f64>());
        }
        let est = p2.quantile();
        assert!(
            (est - 0.9).abs() < 0.02,
            "p90 estimate off: got {est}, expected ~0.9"
        );
    }

    #[test]
    fn p2_quantile_empty_returns_zero() {
        let p2 = P2Quantile::new(0.5);
        assert_eq!(p2.quantile(), 0.0);
    }
}
