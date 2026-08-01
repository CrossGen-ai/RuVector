//! Standard product quantization — Lloyd's k-means per subspace (baseline).
//!
//! This is the classical Jégou/Douze/Schmid PQ, restricted to `k = 256`
//! centroids per subspace (u8 codes). Query-time scoring uses the standard
//! precomputed inner-product LUT.

use crate::{Hit, Lcg, Pq, PqConfig};

/// Standard PQ codebook: `m` subspaces, `256 × d_sub` centroids each.
#[derive(Debug, Clone)]
pub struct StandardPq {
    cfg: PqConfig,
    d_sub: usize,
    /// Row-major: `centroids[m * 256 * d_sub + j * d_sub + t]`.
    centroids: Vec<f32>,
    trained: bool,
}

impl StandardPq {
    pub fn new(cfg: PqConfig) -> Self {
        assert!(cfg.dim % cfg.m == 0, "dim must be divisible by m");
        assert_eq!(cfg.k, 256, "this crate uses 256 centroids per subspace");
        let d_sub = cfg.dim / cfg.m;
        Self {
            cfg,
            d_sub,
            centroids: vec![0.0; cfg.m * cfg.k * d_sub],
            trained: false,
        }
    }

    #[inline]
    fn sub(&self, m: usize) -> &[f32] {
        let start = m * self.cfg.k * self.d_sub;
        &self.centroids[start..start + self.cfg.k * self.d_sub]
    }

    #[inline]
    fn sub_mut(&mut self, m: usize) -> &mut [f32] {
        let start = m * self.cfg.k * self.d_sub;
        let k = self.cfg.k * self.d_sub;
        &mut self.centroids[start..start + k]
    }

    fn train_subspace(&mut self, m: usize, data: &[f32], n: usize) {
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let k = self.cfg.k;

        // Initialize centroids from random distinct rows.
        let mut rng = Lcg(self.cfg.seed.wrapping_add(m as u64 * 0x9E37_79B9));
        {
            let sub = self.sub_mut(m);
            for j in 0..k {
                let idx = (rng.next_u64() as usize) % n;
                let src = &data[idx * d + m * ds..idx * d + m * ds + ds];
                sub[j * ds..(j + 1) * ds].copy_from_slice(src);
            }
        }

        let mut assignments = vec![0u16; n];
        for _iter in 0..self.cfg.iterations {
            // Assignment.
            {
                let sub = self.sub(m);
                for i in 0..n {
                    let x = &data[i * d + m * ds..i * d + m * ds + ds];
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for j in 0..k {
                        let c = &sub[j * ds..(j + 1) * ds];
                        let mut d2 = 0.0f32;
                        for t in 0..ds {
                            let diff = x[t] - c[t];
                            d2 += diff * diff;
                        }
                        if d2 < best_d {
                            best_d = d2;
                            best = j;
                        }
                    }
                    assignments[i] = best as u16;
                }
            }
            // Update = plain mean.
            let mut sums = vec![0.0f64; k * ds];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let j = assignments[i] as usize;
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                for t in 0..ds {
                    sums[j * ds + t] += x[t] as f64;
                }
                counts[j] += 1;
            }
            let sub = self.sub_mut(m);
            for j in 0..k {
                if counts[j] > 0 {
                    let inv = 1.0 / counts[j] as f64;
                    for t in 0..ds {
                        sub[j * ds + t] = (sums[j * ds + t] * inv) as f32;
                    }
                } else {
                    // Re-seed empty centroid from a random point.
                    let idx = (rng.next_u64() as usize) % n;
                    let src = &data[idx * d + m * ds..idx * d + m * ds + ds];
                    sub[j * ds..(j + 1) * ds].copy_from_slice(src);
                }
            }
        }
    }
}

impl Pq for StandardPq {
    fn name(&self) -> &str { "StandardPQ" }

    fn train(&mut self, data: &[f32], n: usize) {
        for m in 0..self.cfg.m {
            self.train_subspace(m, data, n);
        }
        self.trained = true;
    }

    fn encode(&self, data: &[f32], n: usize) -> Vec<u8> {
        assert!(self.trained, "call train() first");
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let m_ = self.cfg.m;
        let mut codes = vec![0u8; n * m_];
        for i in 0..n {
            for m in 0..m_ {
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                let sub = self.sub(m);
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for j in 0..self.cfg.k {
                    let c = &sub[j * ds..(j + 1) * ds];
                    let mut d2 = 0.0f32;
                    for t in 0..ds {
                        let diff = x[t] - c[t];
                        d2 += diff * diff;
                    }
                    if d2 < best_d {
                        best_d = d2;
                        best = j as u8;
                    }
                }
                codes[i * m_ + m] = best;
            }
        }
        codes
    }

    fn search(&self, query: &[f32], codes: &[u8], n: usize, k: usize) -> Vec<Hit> {
        assert_eq!(query.len(), self.cfg.dim);
        let m_ = self.cfg.m;
        let ds = self.d_sub;
        let kc = self.cfg.k;

        // LUT[m * kc + j] = <q_m, c_{m,j}>
        let mut lut = vec![0.0f32; m_ * kc];
        for m in 0..m_ {
            let q_m = &query[m * ds..(m + 1) * ds];
            let sub = self.sub(m);
            for j in 0..kc {
                let c = &sub[j * ds..(j + 1) * ds];
                let mut s = 0.0f32;
                for t in 0..ds {
                    s += q_m[t] * c[t];
                }
                lut[m * kc + j] = s;
            }
        }

        let mut scores: Vec<Hit> = (0..n)
            .map(|i| {
                let base = i * m_;
                let mut s = 0.0f32;
                for m in 0..m_ {
                    s += lut[m * kc + codes[base + m] as usize];
                }
                Hit { id: i as u32, score: s }
            })
            .collect();
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    fn reconstruction_error(&self, data: &[f32], n: usize) -> f64 {
        let d = self.cfg.dim;
        let ds = self.d_sub;
        let m_ = self.cfg.m;
        let mut acc = 0.0f64;
        for i in 0..n {
            for m in 0..m_ {
                let x = &data[i * d + m * ds..i * d + m * ds + ds];
                let sub = self.sub(m);
                let mut best = 0.0f32;
                let mut best_d = f32::INFINITY;
                for j in 0..self.cfg.k {
                    let c = &sub[j * ds..(j + 1) * ds];
                    let mut d2 = 0.0f32;
                    for t in 0..ds {
                        let diff = x[t] - c[t];
                        d2 += diff * diff;
                    }
                    if d2 < best_d {
                        best_d = d2;
                        best = d2;
                    }
                }
                acc += best as f64;
            }
        }
        acc / n as f64
    }

    fn memory_bytes(&self) -> usize {
        self.centroids.len() * std::mem::size_of::<f32>()
    }
}
