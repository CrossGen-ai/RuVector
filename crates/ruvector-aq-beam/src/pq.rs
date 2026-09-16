//! Product Quantization baseline.
//!
//! Splits R^d into M equal sub-spaces (D = d / M), runs k-means with K=256
//! centroids in each, encodes each sub-vector to a single byte, and does
//! ADC search via a per-subspace lookup table of squared distances.

use crate::kmeans::{assign, kmeans};
use crate::{sq_l2, AqError, Quantizer};

pub struct PqParams {
    pub m: usize,
    pub k: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Default for PqParams {
    fn default() -> Self {
        Self { m: 8, k: 256, kmeans_iters: 12, seed: 0xC0FFEE }
    }
}

pub struct PqIndex {
    d: usize,
    m: usize,
    /// centroids[sub][k] = R^{d/m}
    centroids: Vec<Vec<Vec<f32>>>,
}

impl PqIndex {
    pub fn train(data: &[Vec<f32>], p: PqParams) -> Result<Self, AqError> {
        if data.is_empty() {
            return Err(AqError::EmptyTraining);
        }
        let d = data[0].len();
        if d % p.m != 0 {
            return Err(AqError::Dimension { expected: p.m, got: d });
        }
        if p.k == 0 || p.k > 256 || !p.k.is_power_of_two() {
            return Err(AqError::InvalidK);
        }
        let sub_d = d / p.m;
        let mut centroids: Vec<Vec<Vec<f32>>> = Vec::with_capacity(p.m);
        for sub in 0..p.m {
            let subvecs: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v[sub * sub_d..(sub + 1) * sub_d].to_vec())
                .collect();
            let c = kmeans(&subvecs, p.k, sub_d, p.kmeans_iters, p.seed ^ sub as u64);
            centroids.push(c);
        }
        Ok(Self { d, m: p.m, centroids })
    }

    fn sub_d(&self) -> usize {
        self.d / self.m
    }

    fn encode_one(&self, v: &[f32]) -> Vec<u8> {
        let sd = self.sub_d();
        let mut out = vec![0u8; self.m];
        for sub in 0..self.m {
            let sv = &v[sub * sd..(sub + 1) * sd];
            let cs = &self.centroids[sub];
            let a = assign(&[sv.to_vec()], cs);
            out[sub] = a[0];
        }
        out
    }
}

impl Quantizer for PqIndex {
    fn dim(&self) -> usize { self.d }
    fn code_bytes(&self) -> usize { self.m }
    fn name(&self) -> &'static str { "PQ (M=8, K=256)" }

    fn encode_dataset(&self, xs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        xs.iter().map(|v| self.encode_one(v)).collect()
    }

    fn reconstruct(&self, code: &[u8]) -> Vec<f32> {
        let sd = self.sub_d();
        let mut out = vec![0.0f32; self.d];
        for sub in 0..self.m {
            let c = &self.centroids[sub][code[sub] as usize];
            out[sub * sd..(sub + 1) * sd].copy_from_slice(c);
        }
        out
    }

    fn search(&self, query: &[f32], codes: &[Vec<u8>], k: usize) -> Vec<(usize, f32)> {
        let sd = self.sub_d();
        // Precompute per-subspace L2² table: table[sub][c] = ‖q_sub − c‖².
        let mut table: Vec<Vec<f32>> = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let qsub = &query[sub * sd..(sub + 1) * sd];
            let row: Vec<f32> = self.centroids[sub].iter().map(|c| sq_l2(qsub, c)).collect();
            table.push(row);
        }
        let mut scored: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let mut acc = 0.0f32;
                for sub in 0..self.m {
                    acc += table[sub][code[sub] as usize];
                }
                (i, acc)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
