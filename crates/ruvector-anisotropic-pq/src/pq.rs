//! Vanilla Product Quantization baseline.
//!
//! For each of `m` subspaces we run `k`-means (Lloyd's algorithm) on the
//! sub-vectors and store the resulting codebook. Encoding a vector becomes
//! `m` nearest-centroid lookups. This is the textbook PQ from Jégou et al.,
//! "Product Quantization for Nearest Neighbor Search" (PAMI 2011).

use crate::{QuantError, Quantizer};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct Pq {
    dim: usize,
    m: usize,
    k: usize,
    sub_dim: usize,
    /// Codebooks: m * k centroids, each of length sub_dim.
    /// Flat layout: codebook[s][c * sub_dim .. (c+1) * sub_dim]
    pub(crate) codebooks: Vec<Vec<f32>>,
}

impl Pq {
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        max_iter: usize,
        seed: u64,
    ) -> Result<Self, QuantError> {
        if data.is_empty() {
            return Err(QuantError::EmptyTraining);
        }
        if !(1..=256).contains(&k) {
            return Err(QuantError::InvalidK { k });
        }
        let dim = data[0].len();
        if dim % m != 0 {
            return Err(QuantError::SubspaceMismatch { dim, m });
        }
        let sub_dim = dim / m;
        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            let cb = train_kmeans_subspace(data, s, sub_dim, k, max_iter, seed.wrapping_add(s as u64));
            codebooks.push(cb);
        }
        Ok(Self { dim, m, k, sub_dim, codebooks })
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn sub_dim(&self) -> usize { self.sub_dim }
}

/// Standard (unweighted) k-means on subspace `s`.
pub(crate) fn train_kmeans_subspace(
    data: &[Vec<f32>],
    s: usize,
    sub_dim: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(seed);
    // k-means++ style: pick k random distinct points.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    let mut centroids: Vec<f32> = Vec::with_capacity(k * sub_dim);
    for &i in idx.iter().take(k) {
        let v = &data[i][s * sub_dim..(s + 1) * sub_dim];
        centroids.extend_from_slice(v);
    }
    // If fewer than k points, pad by duplicating the last centroid.
    while centroids.len() < k * sub_dim {
        let last_start = centroids.len() - sub_dim;
        let copy: Vec<f32> = centroids[last_start..last_start + sub_dim].to_vec();
        centroids.extend_from_slice(&copy);
    }

    let mut assign = vec![0u32; n];
    for _iter in 0..max_iter {
        // Assignment step.
        let mut moved = 0usize;
        for (i, v) in data.iter().enumerate() {
            let sv = &v[s * sub_dim..(s + 1) * sub_dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cv = &centroids[c * sub_dim..(c + 1) * sub_dim];
                let d = sq_l2(sv, cv);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assign[i] != best as u32 {
                moved += 1;
                assign[i] = best as u32;
            }
        }
        // Update step.
        let mut sums = vec![0f32; k * sub_dim];
        let mut counts = vec![0u32; k];
        for (i, v) in data.iter().enumerate() {
            let c = assign[i] as usize;
            let sv = &v[s * sub_dim..(s + 1) * sub_dim];
            for j in 0..sub_dim {
                sums[c * sub_dim + j] += sv[j];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..sub_dim {
                    centroids[c * sub_dim + j] = sums[c * sub_dim + j] * inv;
                }
            }
        }
        if moved == 0 { break; }
    }
    centroids
}

#[inline]
pub(crate) fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

impl Quantizer for Pq {
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.m];
        for s in 0..self.m {
            let sv = &v[s * self.sub_dim..(s + 1) * self.sub_dim];
            let cb = &self.codebooks[s];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cv = &cb[c * self.sub_dim..(c + 1) * self.sub_dim];
                let d = sq_l2(sv, cv);
                if d < best_d {
                    best_d = d;
                    best = c as u8;
                }
            }
            code[s] = best;
        }
        code
    }

    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for s in 0..self.m {
            let c = code[s] as usize;
            let cv = &self.codebooks[s][c * self.sub_dim..(c + 1) * self.sub_dim];
            v[s * self.sub_dim..(s + 1) * self.sub_dim].copy_from_slice(cv);
        }
        v
    }

    fn m(&self) -> usize { self.m }
    fn k(&self) -> usize { self.k }

    fn dot_table(&self, q: &[f32]) -> Vec<f32> {
        let mut t = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let qs = &q[s * self.sub_dim..(s + 1) * self.sub_dim];
            let cb = &self.codebooks[s];
            for c in 0..self.k {
                let cv = &cb[c * self.sub_dim..(c + 1) * self.sub_dim];
                t[s * self.k + c] = dot(qs, cv);
            }
        }
        t
    }
}
