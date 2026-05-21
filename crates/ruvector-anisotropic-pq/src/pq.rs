//! Plain Product Quantization (Jégou, Douze, Schmid, TPAMI 2011).

use crate::error::Error;
use crate::kmeans::lloyd;
use crate::metrics::sq_l2;
use crate::Quantizer;

#[derive(Clone)]
pub struct Pq {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
    /// codebooks[s][c] is the c-th centroid for subspace s, length = sub_dim.
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl Pq {
    pub fn train(train: &[Vec<f32>], m: usize, k: usize, iters: usize, seed: u64) -> Result<Self, Error> {
        if train.is_empty() {
            return Err(Error::NotEnoughTrain { n: 0, k });
        }
        let d = train[0].len();
        if d % m != 0 {
            return Err(Error::DimNotDivisible { dim: d, m });
        }
        if !(1..=256).contains(&k) {
            return Err(Error::BadK { k });
        }
        if train.len() < k {
            return Err(Error::NotEnoughTrain { n: train.len(), k });
        }
        let sub_dim = d / m;
        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            let sub: Vec<Vec<f32>> = train
                .iter()
                .map(|v| v[s * sub_dim..(s + 1) * sub_dim].to_vec())
                .collect();
            let r = lloyd(&sub, None, k, iters, seed.wrapping_add(s as u64));
            codebooks.push(r.centroids);
        }
        Ok(Self { d, m, k, sub_dim, codebooks })
    }

    /// Per-subspace lookup table for asymmetric scoring.
    pub fn lookup_table(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let q = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            for c in 0..self.k {
                lut[s * self.k + c] = sq_l2(q, &self.codebooks[s][c]);
            }
        }
        lut
    }
}

impl Quantizer for Pq {
    fn encode(&self, x: &[f32]) -> Vec<u8> {
        debug_assert_eq!(x.len(), self.d);
        let mut code = vec![0u8; self.m];
        for s in 0..self.m {
            let sub = &x[s * self.sub_dim..(s + 1) * self.sub_dim];
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let d = sq_l2(sub, &self.codebooks[s][c]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            code[s] = best as u8;
        }
        code
    }

    fn asymmetric_score(&self, query: &[f32], code: &[u8]) -> f32 {
        let lut = self.lookup_table(query);
        let mut s = 0f32;
        for sub in 0..self.m {
            s += lut[sub * self.k + code[sub] as usize];
        }
        s
    }

    fn code_bytes(&self) -> usize {
        self.m
    }

    fn shape(&self) -> (usize, usize, usize) {
        (self.m, self.d, self.k)
    }
}
