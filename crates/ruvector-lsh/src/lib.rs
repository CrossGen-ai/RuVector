//! # ruvector-lsh
//!
//! Bitsampling SimHash multi-probe Locality-Sensitive Hashing for cosine-similarity
//! approximate nearest-neighbor search.
//!
//! ## Design
//!
//! The crate exposes a swappable [`AnnIndex`] trait so callers can drop in
//! different backends. Three reference implementations ship:
//!
//! 1. [`BruteForce`]       - exact cosine baseline.
//! 2. [`SimHashLsh`]       - single-table SimHash (b bits, hamming bucketing).
//! 3. [`MultiProbeSimHash`]- L tables x bitflip-probe variations.
//!
//! ## Memory math
//!
//! * Stored vectors: `n * d * 4` bytes (f32).
//! * Per table:      `n * ceil(b/64) * 8` bytes for packed signatures
//!                 + hashmap overhead (`~32 n` bytes amortized).
//! * MultiProbe L tables: multiplies the per-table cost by L.
//!
//! For `n = 50_000`, `d = 128`, `b = 64`, `L = 4` the index footprint is roughly
//! `50_000 * 128 * 4 = 25.6 MB` vectors + `50_000 * 8 * 4 = 1.6 MB` signatures
//! + ~6.4 MB bucket overhead = ~33.6 MB total.

#![forbid(unsafe_code)]

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashMap;

/// Public trait so callers can mix exact and approximate backends.
pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Brute-force baseline
// ---------------------------------------------------------------------------

pub struct BruteForce {
    pub data: Vec<Vec<f32>>,
    pub dim: usize,
}

impl BruteForce {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let dim = data.first().map(|v| v.len()).unwrap_or(0);
        Self { data, dim }
    }
}

impl AnnIndex for BruteForce {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(query, v)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &'static str {
        "BruteForce"
    }
}

// ---------------------------------------------------------------------------
// 2. SimHash LSH (single table)
// ---------------------------------------------------------------------------

/// SimHash projection: b random Gaussian hyperplanes; bit_i = sign(<r_i, x>).
/// Signature stored as a Vec<u64> for fast hamming via xor+popcnt.
pub struct SimHashProjection {
    pub bits: usize,
    /// hyperplanes laid out row-major (bits x dim) for cache friendliness.
    planes: Vec<f32>,
    dim: usize,
}

impl SimHashProjection {
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut planes = Vec::with_capacity(bits * dim);
        for _ in 0..(bits * dim) {
            let x: f64 = StandardNormal.sample(&mut rng);
            planes.push(x as f32);
        }
        Self { bits, planes, dim }
    }

    pub fn hash(&self, x: &[f32]) -> Vec<u64> {
        let words = self.bits.div_ceil(64);
        let mut out = vec![0u64; words];
        for b in 0..self.bits {
            let row = &self.planes[b * self.dim..(b + 1) * self.dim];
            if dot(row, x) >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
        out
    }
}

#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut h = 0u32;
    for i in 0..a.len() {
        h += (a[i] ^ b[i]).count_ones();
    }
    h
}

pub struct SimHashLsh {
    proj: SimHashProjection,
    sigs: Vec<Vec<u64>>,
    buckets: HashMap<Vec<u64>, Vec<usize>>,
    data: Vec<Vec<f32>>,
    /// fall back to full hamming scan when the bucket has fewer than `min_bucket` candidates.
    min_bucket: usize,
}

impl SimHashLsh {
    pub fn build(data: Vec<Vec<f32>>, bits: usize, seed: u64) -> Self {
        let dim = data.first().map(|v| v.len()).unwrap_or(0);
        let proj = SimHashProjection::new(dim, bits, seed);
        let mut sigs = Vec::with_capacity(data.len());
        let mut buckets: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
        for (i, v) in data.iter().enumerate() {
            let s = proj.hash(v);
            buckets.entry(s.clone()).or_default().push(i);
            sigs.push(s);
        }
        Self {
            proj,
            sigs,
            buckets,
            data,
            min_bucket: 256,
        }
    }
}

impl AnnIndex for SimHashLsh {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let qs = self.proj.hash(query);
        let mut cands: Vec<usize> = self.buckets.get(&qs).cloned().unwrap_or_default();

        // Fallback: if bucket too small, sort all by hamming.
        if cands.len() < self.min_bucket {
            let mut ranked: Vec<(usize, u32)> = (0..self.sigs.len())
                .map(|i| (i, hamming(&qs, &self.sigs[i])))
                .collect();
            ranked.sort_by_key(|x| x.1);
            cands = ranked
                .into_iter()
                .take(self.min_bucket)
                .map(|x| x.0)
                .collect();
        }

        let mut scored: Vec<(usize, f32)> = cands
            .into_iter()
            .map(|i| (i, cosine(query, &self.data[i])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &'static str {
        "SimHashLsh"
    }
}

// ---------------------------------------------------------------------------
// 3. Multi-probe SimHash (L tables + bitflip probes)
// ---------------------------------------------------------------------------

pub struct MultiProbeSimHash {
    tables: Vec<(SimHashProjection, HashMap<Vec<u64>, Vec<usize>>)>,
    sigs_per_table: Vec<Vec<Vec<u64>>>,
    data: Vec<Vec<f32>>,
    pub probes: usize,
    /// If bucket candidates < ef, augment by ranking all points by summed
    /// hamming distance across tables (a fast cosine surrogate).
    pub ef: usize,
}

impl MultiProbeSimHash {
    pub fn build(data: Vec<Vec<f32>>, bits: usize, num_tables: usize, probes: usize, seed: u64) -> Self {
        let dim = data.first().map(|v| v.len()).unwrap_or(0);
        let mut tables = Vec::with_capacity(num_tables);
        let mut sigs_per_table = Vec::with_capacity(num_tables);
        for t in 0..num_tables {
            let proj = SimHashProjection::new(dim, bits, seed.wrapping_add(t as u64 * 0x9E37_79B9_7F4A_7C15));
            let mut buckets: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
            let mut sigs = Vec::with_capacity(data.len());
            for (i, v) in data.iter().enumerate() {
                let s = proj.hash(v);
                buckets.entry(s.clone()).or_default().push(i);
                sigs.push(s);
            }
            tables.push((proj, buckets));
            sigs_per_table.push(sigs);
        }
        Self {
            tables,
            sigs_per_table,
            data,
            probes,
            ef: 200,
        }
    }

    fn probe_keys(&self, sig: &[u64], bits: usize) -> Vec<Vec<u64>> {
        let mut keys = Vec::with_capacity(self.probes + 1);
        keys.push(sig.to_vec());
        // flip up to `probes` lowest-confidence bits — we approximate this by
        // flipping the first `probes` bits (still produces valid probe set).
        for b in 0..self.probes.min(bits) {
            let mut k = sig.to_vec();
            k[b / 64] ^= 1u64 << (b % 64);
            keys.push(k);
        }
        keys
    }
}

impl AnnIndex for MultiProbeSimHash {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut cand_set = std::collections::HashSet::with_capacity(self.ef.max(256));
        // 1. Bucket lookups across all tables with probe variations.
        let mut query_sigs: Vec<Vec<u64>> = Vec::with_capacity(self.tables.len());
        for (proj, buckets) in self.tables.iter() {
            let q = proj.hash(query);
            for key in self.probe_keys(&q, proj.bits) {
                if let Some(ids) = buckets.get(&key) {
                    for &id in ids {
                        cand_set.insert(id);
                    }
                }
            }
            query_sigs.push(q);
        }

        // 2. Hamming-distance augmentation: if too few candidates, rank all
        //    points by SUM of hamming over tables (cosine surrogate via
        //    Johnson-Lindenstrauss) and append top ef.
        if cand_set.len() < self.ef {
            let n = self.data.len();
            let mut ham_ranked: Vec<(usize, u32)> = (0..n)
                .map(|i| {
                    let mut h = 0u32;
                    for t in 0..self.tables.len() {
                        h += hamming(&query_sigs[t], &self.sigs_per_table[t][i]);
                    }
                    (i, h)
                })
                .collect();
            // partial sort top ef
            let take = self.ef.min(n);
            ham_ranked.select_nth_unstable_by_key(take.saturating_sub(1).min(n - 1), |x| x.1);
            for (i, _) in ham_ranked.into_iter().take(take) {
                cand_set.insert(i);
            }
        }

        // 3. Cosine rerank.
        let mut scored: Vec<(usize, f32)> = cand_set
            .into_iter()
            .map(|i| (i, cosine(query, &self.data[i])))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &'static str {
        "MultiProbeSimHash"
    }
}

// ---------------------------------------------------------------------------
// Recall & latency utilities used by benches and tests
// ---------------------------------------------------------------------------

pub fn recall_at_k(approx: &[(usize, f32)], truth: &[(usize, f32)], k: usize) -> f32 {
    if k == 0 {
        return 1.0;
    }
    let truth_set: std::collections::HashSet<usize> =
        truth.iter().take(k).map(|x| x.0).collect();
    let hits = approx
        .iter()
        .take(k)
        .filter(|(i, _)| truth_set.contains(i))
        .count();
    hits as f32 / k as f32
}

pub fn gen_random_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                let x: f64 = StandardNormal.sample(&mut rng);
                x as f32
            })
            .collect();
        l2_normalize(&mut v);
        data.push(v);
    }
    data
}

pub fn gen_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    gen_random_dataset(n, dim, seed)
}

/// Cluster-structured synthetic embeddings: `num_clusters` Gaussian centers,
/// each point = center + sigma * noise, then L2-normalized.
/// More representative of real text/image embeddings than uniform Gaussians.
pub fn gen_clustered_dataset(
    n: usize,
    dim: usize,
    num_clusters: usize,
    sigma: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    // 1. Centers.
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut c: Vec<f32> = (0..dim)
            .map(|_| {
                let x: f64 = StandardNormal.sample(&mut rng);
                x as f32
            })
            .collect();
        l2_normalize(&mut c);
        centers.push(c);
    }
    // 2. Points.
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % num_clusters];
        let mut v: Vec<f32> = c
            .iter()
            .map(|&x| {
                let noise: f64 = StandardNormal.sample(&mut rng);
                x + sigma * (noise as f32)
            })
            .collect();
        l2_normalize(&mut v);
        data.push(v);
    }
    data
}

/// Queries built by sampling near random points in the corpus (simulates
/// "this query is similar to something in the index" — the LSH-relevant case).
pub fn gen_queries_near(corpus: &[Vec<f32>], n: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dim = corpus[0].len();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let pick = (rand::Rng::gen::<u64>(&mut rng) as usize) % corpus.len();
        let base = &corpus[pick];
        let mut v: Vec<f32> = base
            .iter()
            .map(|&x| {
                let noise: f64 = StandardNormal.sample(&mut rng);
                x + sigma * (noise as f32)
            })
            .collect();
        l2_normalize(&mut v);
        let _ = dim;
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests (real, no mocks)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_self_is_one() {
        let v = vec![0.5_f32, 0.5, 0.5, 0.5];
        let c = cosine(&v, &v);
        assert!((c - 1.0).abs() < 1e-5, "cosine(v,v) = {c}");
    }

    #[test]
    fn simhash_signature_dim() {
        let proj = SimHashProjection::new(16, 64, 7);
        let v = vec![0.1_f32; 16];
        let s = proj.hash(&v);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn brute_force_finds_self() {
        let data = gen_random_dataset(200, 16, 1);
        let bf = BruteForce::new(data.clone());
        let res = bf.search(&data[42], 1);
        assert_eq!(res[0].0, 42);
        assert!((res[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn lsh_self_hits_with_fallback() {
        let data = gen_random_dataset(500, 32, 2);
        let idx = SimHashLsh::build(data.clone(), 64, 9);
        let res = idx.search(&data[100], 1);
        assert_eq!(res[0].0, 100);
    }

    #[test]
    fn multiprobe_self_hits() {
        let data = gen_random_dataset(500, 32, 3);
        let idx = MultiProbeSimHash::build(data.clone(), 32, 4, 4, 11);
        let res = idx.search(&data[256], 1);
        assert_eq!(res[0].0, 256);
    }

    #[test]
    fn recall_metric_correctness() {
        let truth = vec![(1, 0.9), (2, 0.8), (3, 0.7)];
        let approx = vec![(1, 0.9), (4, 0.6), (3, 0.7)];
        let r = recall_at_k(&approx, &truth, 3);
        assert!((r - (2.0 / 3.0)).abs() < 1e-5);
    }
}
