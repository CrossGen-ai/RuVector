//! AB-PQ: Adaptive Bit-Budget Product Quantization
//!
//! Classical Product Quantization (Jegou et al., 2011) splits a d-dim vector
//! into m subvectors and quantizes each with a k-means codebook of 2^b entries,
//! **uniformly** across subvectors. This is suboptimal: the expected squared
//! distortion of subquantizer i under a k-means bound is proportional to
//! `sigma_i^2 / 2^(2*b_i)` where sigma_i^2 is the trace of the subvector
//! covariance. Uniform allocation ignores that some subvectors carry far more
//! energy than others.
//!
//! AB-PQ solves a **water-filling** problem to allocate a fixed total bit
//! budget B = sum(b_i) across subvectors to minimize the sum of per-subvector
//! distortion bounds. High-variance subvectors get more bits; low-variance
//! subvectors get fewer (with an integer-rounded, per-subvector min/max clip).
//!
//! Storage stays byte-aligned: bits per subvector are chosen from
//! {2,3,4,5,6,7,8}. A compact packed-bit code layout is used per-vector.
//!
//! See docs/research/nightly/2026-09-28-adaptive-bit-budget-pq/README.md
//! and docs/adr/ADR-350-adaptive-bit-budget-pq.md.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod water_fill;
pub mod pack;

#[cfg(test)]
mod tests;

pub use water_fill::{allocate_bits, per_sub_variance};

/// AB-PQ configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbPqConfig {
    /// input dimension
    pub d: usize,
    /// number of subvectors (must divide d)
    pub m: usize,
    /// total bit budget across all subvectors (B = m * avg_bits)
    pub total_bits: usize,
    /// minimum bits per subvector (>= 1)
    pub min_bits: u8,
    /// maximum bits per subvector (<= 8)
    pub max_bits: u8,
    /// k-means iterations
    pub kmeans_iters: usize,
    /// seed
    pub seed: u64,
}

impl AbPqConfig {
    pub fn new(d: usize, m: usize, avg_bits: usize) -> Self {
        Self {
            d, m,
            total_bits: m * avg_bits,
            min_bits: 2, max_bits: 8,
            kmeans_iters: 25,
            seed: 0xA1B_C0DE_1234_5678,
        }
    }
}

#[derive(Debug, Error)]
pub enum AbPqError {
    #[error("d ({d}) must be divisible by m ({m})")]
    DimNotDivisible { d: usize, m: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("bit budget infeasible: {reason}")]
    InfeasibleBudget { reason: String },
}

/// A single subquantizer: k-means codebook over one subvector slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subquantizer {
    /// bits used (2..=8)
    pub bits: u8,
    /// codebook: 2^bits rows of length dsub
    pub centroids: Vec<f32>,
    pub dsub: usize,
    /// per-subvector variance (trace(cov)) captured at train time
    pub variance: f32,
}

impl Subquantizer {
    pub fn k(&self) -> usize { 1 << self.bits }
    pub fn centroid(&self, code: u16) -> &[f32] {
        let c = code as usize;
        &self.centroids[c * self.dsub..(c + 1) * self.dsub]
    }
}

/// A trained AB-PQ quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbPq {
    pub cfg: AbPqConfig,
    pub subs: Vec<Subquantizer>,
    /// bits[i] for subquantizer i (redundant with subs[i].bits, kept for fast
    /// path)
    pub bits: Vec<u8>,
    /// prefix sum of bits (for bit-packed code offsets)
    pub bit_offsets: Vec<u32>,
    /// total bits per encoded vector
    pub code_bits: u32,
}

impl AbPq {
    /// Train on a slice of contiguous, row-major f32 vectors (n * d).
    pub fn train(cfg: AbPqConfig, data: &[f32]) -> Result<Self, AbPqError> {
        if cfg.d % cfg.m != 0 {
            return Err(AbPqError::DimNotDivisible { d: cfg.d, m: cfg.m });
        }
        if data.is_empty() { return Err(AbPqError::EmptyTraining); }
        let n = data.len() / cfg.d;
        let dsub = cfg.d / cfg.m;

        // 1. compute per-subvector variance
        let variances = per_sub_variance(data, cfg.d, cfg.m);

        // 2. allocate bits via water-filling
        let bits = allocate_bits(
            &variances,
            cfg.total_bits,
            cfg.min_bits,
            cfg.max_bits,
        ).map_err(|e| AbPqError::InfeasibleBudget { reason: e })?;

        // 3. train one k-means codebook per subvector
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut subs = Vec::with_capacity(cfg.m);
        // scratch subvector storage (n * dsub)
        let mut sub_buf = vec![0f32; n * dsub];
        for i in 0..cfg.m {
            // gather column-strip
            for (row, out) in sub_buf.chunks_mut(dsub).enumerate() {
                let start = row * cfg.d + i * dsub;
                out.copy_from_slice(&data[start..start + dsub]);
            }
            let k = 1usize << bits[i];
            let centroids = kmeans(&sub_buf, dsub, k, cfg.kmeans_iters, &mut rng);
            subs.push(Subquantizer {
                bits: bits[i], centroids, dsub, variance: variances[i],
            });
        }

        let mut bit_offsets = Vec::with_capacity(cfg.m + 1);
        let mut acc: u32 = 0;
        for &b in &bits { bit_offsets.push(acc); acc += b as u32; }
        bit_offsets.push(acc);
        let code_bits = acc;

        Ok(AbPq { cfg, subs, bits, bit_offsets, code_bits })
    }

    /// Bytes per encoded vector (ceiling).
    pub fn code_bytes(&self) -> usize {
        ((self.code_bits + 7) / 8) as usize
    }

    /// Encode one vector into the caller-provided buffer.
    pub fn encode_into(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(x.len(), self.cfg.d);
        debug_assert_eq!(out.len(), self.code_bytes());
        out.fill(0);
        let dsub = self.cfg.d / self.cfg.m;
        for i in 0..self.cfg.m {
            let sub = &self.subs[i];
            let xslice = &x[i * dsub..(i + 1) * dsub];
            let code = nearest_centroid(xslice, sub);
            pack::write_bits(out, self.bit_offsets[i], sub.bits, code as u32);
        }
    }

    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        let mut buf = vec![0u8; self.code_bytes()];
        self.encode_into(x, &mut buf);
        buf
    }

    /// Batch encode a matrix of n * d f32.
    pub fn encode_batch(&self, xs: &[f32]) -> Vec<u8> {
        let n = xs.len() / self.cfg.d;
        let cb = self.code_bytes();
        let mut out = vec![0u8; n * cb];
        for i in 0..n {
            self.encode_into(
                &xs[i * self.cfg.d..(i + 1) * self.cfg.d],
                &mut out[i * cb..(i + 1) * cb],
            );
        }
        out
    }

    /// Build asymmetric distance computation (ADC) table for one query.
    /// Returns a flat table of size sum_i(2^bits[i]) with per-subvector blocks.
    pub fn build_adc_table(&self, q: &[f32]) -> AdcTable {
        debug_assert_eq!(q.len(), self.cfg.d);
        let dsub = self.cfg.d / self.cfg.m;
        let mut table = Vec::with_capacity(
            self.subs.iter().map(|s| s.k()).sum()
        );
        let mut block_offsets = Vec::with_capacity(self.cfg.m + 1);
        let mut acc = 0u32;
        for (i, sub) in self.subs.iter().enumerate() {
            block_offsets.push(acc);
            let qslice = &q[i * dsub..(i + 1) * dsub];
            for c in 0..sub.k() {
                let cent = sub.centroid(c as u16);
                let mut d = 0f32;
                for j in 0..dsub {
                    let diff = qslice[j] - cent[j];
                    d += diff * diff;
                }
                table.push(d);
            }
            acc += sub.k() as u32;
        }
        block_offsets.push(acc);
        AdcTable { table, block_offsets, bit_offsets: self.bit_offsets.clone(), bits: self.bits.clone() }
    }

    /// Estimate squared L2 distance from query to a single encoded vector via ADC.
    pub fn adc_distance(&self, code: &[u8], table: &AdcTable) -> f32 {
        let mut d = 0f32;
        for i in 0..self.cfg.m {
            let idx = pack::read_bits(code, self.bit_offsets[i], self.bits[i]) as usize;
            d += table.table[table.block_offsets[i] as usize + idx];
        }
        d
    }
}

/// Precomputed ADC lookup for one query.
#[derive(Debug, Clone)]
pub struct AdcTable {
    pub table: Vec<f32>,
    pub block_offsets: Vec<u32>,
    pub bit_offsets: Vec<u32>,
    pub bits: Vec<u8>,
}

// ---- helpers ----

fn nearest_centroid(x: &[f32], sub: &Subquantizer) -> u16 {
    let k = sub.k();
    let mut best_c = 0u16;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let cent = sub.centroid(c as u16);
        let mut d = 0f32;
        for j in 0..sub.dsub {
            let diff = x[j] - cent[j];
            d += diff * diff;
        }
        if d < best_d { best_d = d; best_c = c as u16; }
    }
    best_c
}

/// Simple k-means (Lloyd's) for training subquantizer codebooks.
/// data: n rows of dsub. Returns k * dsub centroid table.
fn kmeans(data: &[f32], dsub: usize, k: usize, iters: usize, rng: &mut StdRng) -> Vec<f32> {
    let n = data.len() / dsub;
    if n == 0 || k == 0 { return Vec::new(); }
    let k = k.min(n);
    // k-means++ init: pick first at random, then farthest-point
    let mut centroids = vec![0f32; k * dsub];
    let first = rng.gen_range(0..n);
    centroids[..dsub].copy_from_slice(&data[first * dsub..(first + 1) * dsub]);
    let mut mind = vec![f32::INFINITY; n];
    for c in 1..k {
        // update mind against last centroid
        let prev = &centroids[(c - 1) * dsub..c * dsub];
        for i in 0..n {
            let mut d = 0f32;
            for j in 0..dsub {
                let diff = data[i * dsub + j] - prev[j];
                d += diff * diff;
            }
            if d < mind[i] { mind[i] = d; }
        }
        // sample proportional to mind (D^2)
        let total: f32 = mind.iter().sum();
        let mut pick = n - 1;
        if total > 0.0 {
            let mut r: f32 = rng.gen::<f32>() * total;
            for i in 0..n {
                r -= mind[i];
                if r <= 0.0 { pick = i; break; }
            }
        } else {
            pick = rng.gen_range(0..n);
        }
        centroids[c * dsub..(c + 1) * dsub]
            .copy_from_slice(&data[pick * dsub..(pick + 1) * dsub]);
    }

    let mut assign = vec![0u32; n];
    let mut sums = vec![0f32; k * dsub];
    let mut counts = vec![0u32; k];
    for _ in 0..iters {
        // assign
        for i in 0..n {
            let x = &data[i * dsub..(i + 1) * dsub];
            let mut bc = 0usize;
            let mut bd = f32::INFINITY;
            for c in 0..k {
                let cc = &centroids[c * dsub..(c + 1) * dsub];
                let mut d = 0f32;
                for j in 0..dsub {
                    let diff = x[j] - cc[j];
                    d += diff * diff;
                }
                if d < bd { bd = d; bc = c; }
            }
            assign[i] = bc as u32;
        }
        // update
        sums.iter_mut().for_each(|v| *v = 0.0);
        counts.iter_mut().for_each(|v| *v = 0);
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let dst = &mut sums[c * dsub..(c + 1) * dsub];
            let x = &data[i * dsub..(i + 1) * dsub];
            for j in 0..dsub { dst[j] += x[j]; }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // re-seed empty cluster at farthest point
                let mut fi = 0usize;
                let mut fd = -1f32;
                for i in 0..n {
                    let cc = &centroids[assign[i] as usize * dsub..(assign[i] as usize + 1) * dsub];
                    let x = &data[i * dsub..(i + 1) * dsub];
                    let mut d = 0f32;
                    for j in 0..dsub {
                        let diff = x[j] - cc[j];
                        d += diff * diff;
                    }
                    if d > fd { fd = d; fi = i; }
                }
                centroids[c * dsub..(c + 1) * dsub]
                    .copy_from_slice(&data[fi * dsub..(fi + 1) * dsub]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dsub {
                    centroids[c * dsub + j] = sums[c * dsub + j] * inv;
                }
            }
        }
    }
    centroids
}
