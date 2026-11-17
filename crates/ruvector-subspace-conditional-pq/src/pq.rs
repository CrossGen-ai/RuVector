//! Baseline Product Quantization (PQ) implementation used as the apples-to-apples
//! comparison point for SC-PQ. Both share the same subspace split, K, and training
//! iteration budget, so any recall or MSE delta is attributable to conditioning.

use crate::{codebook::train_kmeans, Codebook, Quantizer};

pub struct Pq {
    d: usize,
    m: usize,
    k: usize,
    sub_d: usize,
    books: Vec<Codebook>,
}

impl Pq {
    pub fn train(data: &[f32], d: usize, m: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(d % m == 0, "D must be divisible by M");
        let sub_d = d / m;
        let n = data.len() / d;
        let mut sub = vec![0f32; n * sub_d];
        let mut books = Vec::with_capacity(m);
        for mi in 0..m {
            for i in 0..n {
                let src = &data[i * d + mi * sub_d..i * d + (mi + 1) * sub_d];
                let dst = &mut sub[i * sub_d..(i + 1) * sub_d];
                dst.copy_from_slice(src);
            }
            books.push(train_kmeans(&sub, sub_d, k, iters, seed.wrapping_add(mi as u64)));
        }
        Self {
            d,
            m,
            k,
            sub_d,
            books,
        }
    }
}

impl Quantizer for Pq {
    fn encode(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.d;
        let mut codes = vec![0u8; n * self.m];
        for i in 0..n {
            for mi in 0..self.m {
                let x = &data[i * self.d + mi * self.sub_d..i * self.d + (mi + 1) * self.sub_d];
                let book = &self.books[mi];
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for j in 0..self.k {
                    let c = book.centroid(j);
                    let mut s = 0f32;
                    for t in 0..self.sub_d {
                        let dx = x[t] - c[t];
                        s += dx * dx;
                    }
                    if s < best_d {
                        best_d = s;
                        best = j as u8;
                    }
                }
                codes[i * self.m + mi] = best;
            }
        }
        codes
    }

    fn build_lut(&self, query: &[f32]) -> Vec<f32> {
        // layout: lut[mi * K + k]
        let mut lut = vec![0f32; self.m * self.k];
        for mi in 0..self.m {
            let q = &query[mi * self.sub_d..(mi + 1) * self.sub_d];
            let book = &self.books[mi];
            for j in 0..self.k {
                let c = book.centroid(j);
                let mut s = 0f32;
                for t in 0..self.sub_d {
                    let dx = q[t] - c[t];
                    s += dx * dx;
                }
                lut[mi * self.k + j] = s;
            }
        }
        lut
    }

    fn score(&self, lut: &[f32], codes: &[u8]) -> Vec<f32> {
        let n = codes.len() / self.m;
        let mut out = vec![0f32; n];
        for i in 0..n {
            let mut s = 0f32;
            for mi in 0..self.m {
                let c = codes[i * self.m + mi] as usize;
                s += lut[mi * self.k + c];
            }
            out[i] = s;
        }
        out
    }

    fn m(&self) -> usize {
        self.m
    }
    fn k(&self) -> usize {
        self.k
    }
    fn d(&self) -> usize {
        self.d
    }
}
