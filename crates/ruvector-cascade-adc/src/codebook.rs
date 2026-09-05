//! Product-quantizer codebook training and code assignment.
//!
//! Two levels of codebook are produced:
//!   * `Codebook8` — the primary PQ trained by Lloyd's algorithm (k-means)
//!     with `K8 = 256` centroids per subspace.
//!   * `Codebook4` — a *coherent* coarsening of `Codebook8` down to
//!     `K4 = 16` centroids per subspace, obtained by another Lloyd pass over
//!     the 256 centroids. Each 8-bit code carries a companion 4-bit code
//!     that indexes the parent centroid — so an 8-bit code and its 4-bit
//!     partner refer to *the same physical vector region*, which is what
//!     makes Stage-1 pruning safe.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Fine centroids per subspace (8-bit).
pub const K8: usize = 256;
/// Coarse centroids per subspace (4-bit).
pub const K4: usize = 16;

/// Configuration for PQ training.
#[derive(Clone, Copy, Debug)]
pub struct TrainingConfig {
    /// Number of k-means iterations for the fine (8-bit) codebook.
    pub iters_fine: usize,
    /// Number of k-means iterations for the coarse (4-bit) codebook.
    pub iters_coarse: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iters_fine: 12,
            iters_coarse: 8,
            seed: 0xC45CADE_ADC,
        }
    }
}

/// Fine product-quantizer codebook (256 centroids per subspace).
#[derive(Clone, Debug)]
pub struct Codebook8 {
    /// Centroids laid out as `m * K8 * dsub` f32s, indexed by
    /// `centroids[(subspace * K8 + code) * dsub + t]`.
    pub centroids: Vec<f32>,
    pub m: usize,
    pub dsub: usize,
}

/// Coarse companion codebook (16 centroids per subspace), coherent with the
/// fine codebook via [`Codebook8`]::coarse_map.
#[derive(Clone, Debug)]
pub struct Codebook4 {
    pub centroids: Vec<f32>, // m * K4 * dsub
    pub m: usize,
    pub dsub: usize,
    /// For each subspace, `parent[j][fine_code]` gives the coarse code that
    /// the fine centroid was assigned to. Length `m * K8`.
    pub parent: Vec<u8>,
}

impl Codebook8 {
    /// Train a fine PQ codebook on `train` (row-major, `n * d`).
    pub fn train(train: &[f32], n: usize, d: usize, m: usize, cfg: &TrainingConfig) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        let dsub = d / m;
        let mut centroids = vec![0.0f32; m * K8 * dsub];
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        // One k-means per subspace, using vectors' subspace projection.
        for j in 0..m {
            // Seed centroids by random rows.
            for k in 0..K8 {
                let row = rng.gen_range(0..n);
                let src = &train[row * d + j * dsub..row * d + j * dsub + dsub];
                let dst_off = (j * K8 + k) * dsub;
                centroids[dst_off..dst_off + dsub].copy_from_slice(src);
            }

            let mut counts = vec![0u32; K8];
            let mut new_centroids = vec![0.0f32; K8 * dsub];

            for _iter in 0..cfg.iters_fine {
                counts.iter_mut().for_each(|c| *c = 0);
                new_centroids.iter_mut().for_each(|v| *v = 0.0);

                for row in 0..n {
                    let src = &train[row * d + j * dsub..row * d + j * dsub + dsub];
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for k in 0..K8 {
                        let c_off = (j * K8 + k) * dsub;
                        let dist = l2_sq(src, &centroids[c_off..c_off + dsub]);
                        if dist < best_d {
                            best_d = dist;
                            best = k;
                        }
                    }
                    counts[best] += 1;
                    let n_off = best * dsub;
                    for t in 0..dsub {
                        new_centroids[n_off + t] += src[t];
                    }
                }

                // Normalise; re-seed empty clusters with a random row.
                for k in 0..K8 {
                    let n_off = k * dsub;
                    let c_off = (j * K8 + k) * dsub;
                    if counts[k] == 0 {
                        let row = rng.gen_range(0..n);
                        let src = &train[row * d + j * dsub..row * d + j * dsub + dsub];
                        centroids[c_off..c_off + dsub].copy_from_slice(src);
                    } else {
                        let inv = 1.0 / counts[k] as f32;
                        for t in 0..dsub {
                            centroids[c_off + t] = new_centroids[n_off + t] * inv;
                        }
                    }
                }
            }
        }

        Self { centroids, m, dsub }
    }

    /// Encode a batch to 8-bit PQ codes (`n * m` bytes).
    pub fn encode(&self, vecs: &[f32], n: usize) -> Vec<u8> {
        let d = self.m * self.dsub;
        let mut codes = vec![0u8; n * self.m];
        for row in 0..n {
            for j in 0..self.m {
                let src = &vecs[row * d + j * self.dsub..row * d + j * self.dsub + self.dsub];
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for k in 0..K8 {
                    let c_off = (j * K8 + k) * self.dsub;
                    let dist = l2_sq(src, &self.centroids[c_off..c_off + self.dsub]);
                    if dist < best_d {
                        best_d = dist;
                        best = k as u8;
                    }
                }
                codes[row * self.m + j] = best;
            }
        }
        codes
    }

    /// Build the coarse (4-bit) companion by k-means over centroids.
    pub fn coarse(&self, cfg: &TrainingConfig) -> Codebook4 {
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xC0A45E);
        let mut coarse = vec![0.0f32; self.m * K4 * self.dsub];
        let mut parent = vec![0u8; self.m * K8];

        for j in 0..self.m {
            // Seed coarse centroids from a subset of fine centroids.
            for k in 0..K4 {
                let idx = rng.gen_range(0..K8);
                let src_off = (j * K8 + idx) * self.dsub;
                let dst_off = (j * K4 + k) * self.dsub;
                coarse[dst_off..dst_off + self.dsub]
                    .copy_from_slice(&self.centroids[src_off..src_off + self.dsub]);
            }

            let mut counts = vec![0u32; K4];
            let mut new_c = vec![0.0f32; K4 * self.dsub];

            for _iter in 0..cfg.iters_coarse {
                counts.iter_mut().for_each(|c| *c = 0);
                new_c.iter_mut().for_each(|v| *v = 0.0);

                for fine in 0..K8 {
                    let f_off = (j * K8 + fine) * self.dsub;
                    let src = &self.centroids[f_off..f_off + self.dsub];
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for k in 0..K4 {
                        let c_off = (j * K4 + k) * self.dsub;
                        let dist = l2_sq(src, &coarse[c_off..c_off + self.dsub]);
                        if dist < best_d {
                            best_d = dist;
                            best = k;
                        }
                    }
                    counts[best] += 1;
                    let n_off = best * self.dsub;
                    for t in 0..self.dsub {
                        new_c[n_off + t] += src[t];
                    }
                    parent[j * K8 + fine] = best as u8;
                }

                for k in 0..K4 {
                    let n_off = k * self.dsub;
                    let c_off = (j * K4 + k) * self.dsub;
                    if counts[k] == 0 {
                        let idx = rng.gen_range(0..K8);
                        let src_off = (j * K8 + idx) * self.dsub;
                        coarse[c_off..c_off + self.dsub]
                            .copy_from_slice(&self.centroids[src_off..src_off + self.dsub]);
                    } else {
                        let inv = 1.0 / counts[k] as f32;
                        for t in 0..self.dsub {
                            coarse[c_off + t] = new_c[n_off + t] * inv;
                        }
                    }
                }
            }
        }

        Codebook4 {
            centroids: coarse,
            m: self.m,
            dsub: self.dsub,
            parent,
        }
    }
}

impl Codebook4 {
    /// Given 8-bit codes (row-major, `n * m`), derive the packed 4-bit codes
    /// (row-major, `n * ceil(m/2)`) by mapping through `parent`.
    pub fn pack_from_fine(&self, fine_codes: &[u8], n: usize) -> Vec<u8> {
        let packed_stride = self.m.div_ceil(2);
        let mut out = vec![0u8; n * packed_stride];
        for row in 0..n {
            for j in 0..self.m {
                let fine = fine_codes[row * self.m + j] as usize;
                let coarse = self.parent[j * K8 + fine];
                let byte_idx = row * packed_stride + j / 2;
                if j % 2 == 0 {
                    out[byte_idx] |= coarse & 0x0F;
                } else {
                    out[byte_idx] |= (coarse & 0x0F) << 4;
                }
            }
        }
        out
    }
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_and_encode_shapes_match() {
        let n = 200;
        let d = 16;
        let m = 4;
        let mut rng = StdRng::seed_from_u64(1);
        let train: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let cb = Codebook8::train(&train, n, d, m, &TrainingConfig::default());
        assert_eq!(cb.centroids.len(), m * K8 * (d / m));
        let codes = cb.encode(&train, n);
        assert_eq!(codes.len(), n * m);

        let coarse = cb.coarse(&TrainingConfig::default());
        assert_eq!(coarse.centroids.len(), m * K4 * (d / m));
        let packed = coarse.pack_from_fine(&codes, n);
        assert_eq!(packed.len(), n * m.div_ceil(2));
    }
}
