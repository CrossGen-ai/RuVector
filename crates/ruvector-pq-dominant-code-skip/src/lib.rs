//! Product Quantization with dominant-code skip bitmap.
//!
//! Standard asymmetric distance computation (ADC) scan over PQ-encoded vectors
//! touches `M` LUT entries per candidate, where `M` is the number of subspaces.
//! This crate reduces that memory traffic by exploiting the fact that, after
//! k-means training on realistic data, the centroid frequency distribution per
//! subspace is far from uniform: one centroid often claims 15–35% of all codes.
//!
//! We rebalance each subspace's codebook so the most frequent centroid lives at
//! index 0 (the *dominant* code). Vectors are then stored as:
//!
//!   - a packed exception bitmap of M bits per vector (one bit per subspace,
//!     set when the code is NOT dominant), plus
//!   - the exception codes themselves, laid out contiguously.
//!
//! At query time the scanner precomputes a per-query constant
//!   `C0 = sum_{m=0..M} LUT[m][0]`
//! and, for each candidate, adds `C0` plus only the delta `LUT[m][c] - LUT[m][0]`
//! for each non-dominant subspace. Subspaces that stayed dominant contribute no
//! memory touches at all. The approach is exact: no recall loss versus baseline
//! PQ, only reduced wall-clock from lower bandwidth.
//!
//! No `unsafe`, no deps, deterministic PRNG for reproducible training.

#![forbid(unsafe_code)]

use std::time::Instant;

// ---------------------------------------------------------------------------
// Deterministic xorshift64 PRNG (reproducible across runs; no deps).
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-9).min(1.0 - 1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    #[inline]
    pub fn range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }
}

// ---------------------------------------------------------------------------
// Basic L2.
// ---------------------------------------------------------------------------

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

// ---------------------------------------------------------------------------
// Dataset holder.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Dataset {
    pub dim: usize,
    pub vectors: Vec<f32>, // n * dim row-major
    pub n: usize,
}

impl Dataset {
    pub fn new(n: usize, dim: usize) -> Self {
        Self {
            dim,
            vectors: vec![0.0; n * dim],
            n,
        }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let s = i * self.dim;
        &self.vectors[s..s + self.dim]
    }
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let s = i * self.dim;
        &mut self.vectors[s..s + self.dim]
    }
}

// ---------------------------------------------------------------------------
// PQ codebook: M subspaces, K centroids each, each of dim_sub = dim/M.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Codebook {
    pub m: usize,                // subspaces
    pub k: usize,                // centroids per subspace (K<=256 so u8 codes fit)
    pub dim_sub: usize,          // sub-vector dim
    pub centroids: Vec<Vec<f32>>, // centroids[m] = K*dim_sub row-major
}

impl Codebook {
    pub fn train(data: &Dataset, m: usize, k: usize, iters: usize, seed: u64) -> Self {
        assert!(k <= 256, "K must fit in u8");
        assert_eq!(data.dim % m, 0, "dim must be divisible by M");
        let dim_sub = data.dim / m;
        let mut rng = Rng::new(seed);
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(m);
        for sub in 0..m {
            // k-means++ init for stability.
            let mut c = vec![0.0f32; k * dim_sub];
            // first centroid: random sample.
            let first = rng.range(data.n);
            let off = first * data.dim + sub * dim_sub;
            c[..dim_sub].copy_from_slice(&data.vectors[off..off + dim_sub]);
            // remaining centroids: D^2-weighted.
            let mut min_d = vec![f32::INFINITY; data.n];
            for ki in 1..k {
                // update min_d against newly added centroid (ki-1).
                let cstart = (ki - 1) * dim_sub;
                for i in 0..data.n {
                    let voff = i * data.dim + sub * dim_sub;
                    let v = &data.vectors[voff..voff + dim_sub];
                    let d = l2_sq(v, &c[cstart..cstart + dim_sub]);
                    if d < min_d[i] {
                        min_d[i] = d;
                    }
                }
                // sample proportional to min_d.
                let total: f32 = min_d.iter().copied().sum::<f32>().max(1e-9);
                let target = rng.next_f32() * total;
                let mut acc = 0.0f32;
                let mut chosen = data.n - 1;
                for i in 0..data.n {
                    acc += min_d[i];
                    if acc >= target {
                        chosen = i;
                        break;
                    }
                }
                let off = chosen * data.dim + sub * dim_sub;
                c[ki * dim_sub..(ki + 1) * dim_sub]
                    .copy_from_slice(&data.vectors[off..off + dim_sub]);
            }
            // Lloyd iterations.
            let mut assign = vec![0u16; data.n];
            for _ in 0..iters {
                // assign
                for i in 0..data.n {
                    let voff = i * data.dim + sub * dim_sub;
                    let v = &data.vectors[voff..voff + dim_sub];
                    let mut best = 0u16;
                    let mut bd = f32::INFINITY;
                    for ki in 0..k {
                        let d = l2_sq(v, &c[ki * dim_sub..(ki + 1) * dim_sub]);
                        if d < bd {
                            bd = d;
                            best = ki as u16;
                        }
                    }
                    assign[i] = best;
                }
                // update
                let mut sums = vec![0.0f32; k * dim_sub];
                let mut counts = vec![0u32; k];
                for i in 0..data.n {
                    let ki = assign[i] as usize;
                    counts[ki] += 1;
                    let voff = i * data.dim + sub * dim_sub;
                    for d in 0..dim_sub {
                        sums[ki * dim_sub + d] += data.vectors[voff + d];
                    }
                }
                for ki in 0..k {
                    if counts[ki] == 0 {
                        // reseed empty cluster with random point
                        let r = rng.range(data.n);
                        let voff = r * data.dim + sub * dim_sub;
                        c[ki * dim_sub..(ki + 1) * dim_sub]
                            .copy_from_slice(&data.vectors[voff..voff + dim_sub]);
                    } else {
                        let inv = 1.0 / counts[ki] as f32;
                        for d in 0..dim_sub {
                            c[ki * dim_sub + d] = sums[ki * dim_sub + d] * inv;
                        }
                    }
                }
            }
            centroids.push(c);
        }
        Self { m, k, dim_sub, centroids }
    }

    /// Encode a single vector: returns M code bytes.
    pub fn encode_one(&self, v: &[f32]) -> Vec<u8> {
        let mut codes = vec![0u8; self.m];
        for sub in 0..self.m {
            let sv = &v[sub * self.dim_sub..(sub + 1) * self.dim_sub];
            let c = &self.centroids[sub];
            let mut best = 0u8;
            let mut bd = f32::INFINITY;
            for ki in 0..self.k {
                let d = l2_sq(sv, &c[ki * self.dim_sub..(ki + 1) * self.dim_sub]);
                if d < bd {
                    bd = d;
                    best = ki as u8;
                }
            }
            codes[sub] = best;
        }
        codes
    }

    /// Build the per-query LUT: `lut[m*K + ki]` = L2(query_sub_m, centroid_m_ki).
    pub fn lut(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0f32; self.m * self.k];
        for sub in 0..self.m {
            let qv = &query[sub * self.dim_sub..(sub + 1) * self.dim_sub];
            let c = &self.centroids[sub];
            for ki in 0..self.k {
                lut[sub * self.k + ki] =
                    l2_sq(qv, &c[ki * self.dim_sub..(ki + 1) * self.dim_sub]);
            }
        }
        lut
    }
}

// ---------------------------------------------------------------------------
// Baseline PQ index: flat M-byte-per-vector codes.
// ---------------------------------------------------------------------------

pub struct BaselinePq {
    pub codebook: Codebook,
    pub codes: Vec<u8>, // n * M row-major
    pub n: usize,
}

impl BaselinePq {
    pub fn build(data: &Dataset, codebook: Codebook) -> Self {
        let m = codebook.m;
        let mut codes = vec![0u8; data.n * m];
        for i in 0..data.n {
            let c = codebook.encode_one(data.row(i));
            codes[i * m..(i + 1) * m].copy_from_slice(&c);
        }
        Self { codebook, codes, n: data.n }
    }

    /// ADC scan: returns Vec<(id, dist)> for the k nearest.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = self.codebook.lut(query);
        let m = self.codebook.m;
        let kcent = self.codebook.k;
        let mut dists = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let row = &self.codes[i * m..(i + 1) * m];
            let mut d = 0.0f32;
            for sub in 0..m {
                d += lut[sub * kcent + row[sub] as usize];
            }
            dists.push((i as u32, d));
        }
        topk(&mut dists, k);
        dists.truncate(k);
        dists
    }
}

// ---------------------------------------------------------------------------
// Dominant-code-skip PQ index.
// ---------------------------------------------------------------------------

pub struct DominantSkipPq {
    pub codebook: Codebook,
    /// bitmap[(i * m_bytes)..] holds M bits per vector; bit=1 means non-dominant.
    pub bitmap: Vec<u8>,
    pub m_bits: usize,  // = codebook.m
    pub m_bytes: usize, // bits rounded up to bytes
    /// exceptions laid out per vector: offsets[i]..offsets[i+1] indexes into exc_codes.
    pub offsets: Vec<u32>,
    pub exc_codes: Vec<u8>,
    /// Permutation: `perm[m][new_index] = old_index`. Used to remap LUT.
    pub perm: Vec<Vec<u8>>, // M * K
    pub n: usize,
}

impl DominantSkipPq {
    /// Build from an existing BaselinePq by rebalancing codebooks so the
    /// most frequent centroid per subspace lives at index 0.
    pub fn build(data: &Dataset, mut codebook: Codebook) -> Self {
        let m = codebook.m;
        let k = codebook.k;
        // First encode once with original codebook to collect frequencies.
        let mut freq = vec![vec![0u32; k]; m];
        let mut raw_codes = vec![0u8; data.n * m];
        for i in 0..data.n {
            let c = codebook.encode_one(data.row(i));
            raw_codes[i * m..(i + 1) * m].copy_from_slice(&c);
            for sub in 0..m {
                freq[sub][c[sub] as usize] += 1;
            }
        }
        // Compute permutation per subspace: new 0 = argmax freq.
        let mut perm: Vec<Vec<u8>> = Vec::with_capacity(m);
        for sub in 0..m {
            let mut idx: Vec<u8> = (0..k as u8).collect();
            // find dominant
            let mut dom = 0usize;
            let mut dmax = 0u32;
            for ki in 0..k {
                if freq[sub][ki] > dmax {
                    dmax = freq[sub][ki];
                    dom = ki;
                }
            }
            // swap dom <-> 0 in idx, and swap centroid rows accordingly.
            idx.swap(0, dom);
            let c = &mut codebook.centroids[sub];
            if dom != 0 {
                let dim_sub = codebook.dim_sub;
                let (lo, hi) = c.split_at_mut(dom * dim_sub);
                let a = &mut lo[..dim_sub];
                let b = &mut hi[..dim_sub];
                for d in 0..dim_sub {
                    std::mem::swap(&mut a[d], &mut b[d]);
                }
            }
            perm.push(idx);
        }
        // Re-encode: a code that WAS `dom` becomes 0, code that was 0 becomes dom,
        // others unchanged. Equivalent to applying the swap permutation.
        for i in 0..data.n {
            for sub in 0..m {
                let c = raw_codes[i * m + sub];
                let dom = perm[sub][0]; // original index of new 0
                let new_c = if c == 0 {
                    dom
                } else if c == dom {
                    0
                } else {
                    c
                };
                raw_codes[i * m + sub] = new_c;
            }
        }
        // Build bitmap + exceptions.
        let m_bits = m;
        let m_bytes = (m_bits + 7) / 8;
        let mut bitmap = vec![0u8; data.n * m_bytes];
        let mut offsets = Vec::with_capacity(data.n + 1);
        let mut exc_codes: Vec<u8> = Vec::with_capacity(data.n * m / 4);
        offsets.push(0u32);
        for i in 0..data.n {
            let bm_off = i * m_bytes;
            for sub in 0..m {
                let c = raw_codes[i * m + sub];
                if c != 0 {
                    bitmap[bm_off + sub / 8] |= 1 << (sub & 7);
                    exc_codes.push(c);
                }
            }
            offsets.push(exc_codes.len() as u32);
        }
        Self {
            codebook,
            bitmap,
            m_bits,
            m_bytes,
            offsets,
            exc_codes,
            perm,
            n: data.n,
        }
    }

    /// ADC scan using dominant-code skip.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = self.codebook.lut(query);
        let m = self.codebook.m;
        let kcent = self.codebook.k;
        // Precompute per-query constant = sum of LUT[m][0].
        let mut c0 = 0.0f32;
        for sub in 0..m {
            c0 += lut[sub * kcent];
        }
        let mut dists = Vec::with_capacity(self.n);
        let m_bytes = self.m_bytes;
        for i in 0..self.n {
            let bm_off = i * m_bytes;
            let exc_start = self.offsets[i] as usize;
            let exc_end = self.offsets[i + 1] as usize;
            let exc = &self.exc_codes[exc_start..exc_end];
            let mut d = c0;
            let mut xi = 0usize;
            // Iterate exception bits only. This avoids per-subspace LUT touches
            // for the dominant positions.
            for byte_i in 0..m_bytes {
                let mut b = self.bitmap[bm_off + byte_i];
                while b != 0 {
                    let bit = b.trailing_zeros() as usize;
                    let sub = byte_i * 8 + bit;
                    if sub < m {
                        let code = exc[xi] as usize;
                        xi += 1;
                        d += lut[sub * kcent + code] - lut[sub * kcent];
                    }
                    b &= b - 1;
                }
            }
            dists.push((i as u32, d));
        }
        topk(&mut dists, k);
        dists.truncate(k);
        dists
    }

    /// Compactness statistics.
    pub fn stats(&self) -> SkipStats {
        let total_bits = self.n * self.m_bits;
        let set_bits = self.exc_codes.len();
        let skip_rate = 1.0 - (set_bits as f64) / (total_bits as f64);
        let bytes_baseline = (self.n * self.m_bits) as u64;
        let bytes_this = (self.bitmap.len()
            + self.exc_codes.len()
            + self.offsets.len() * 4) as u64;
        SkipStats {
            skip_rate,
            bytes_baseline,
            bytes_skip: bytes_this,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SkipStats {
    pub skip_rate: f64,
    pub bytes_baseline: u64,
    pub bytes_skip: u64,
}

// ---------------------------------------------------------------------------
// Brute force for ground truth.
// ---------------------------------------------------------------------------

pub fn brute_force(data: &Dataset, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = (0..data.n)
        .map(|i| (i as u32, l2_sq(data.row(i), query)))
        .collect();
    topk(&mut dists, k);
    dists.truncate(k);
    dists
}

fn topk(xs: &mut Vec<(u32, f32)>, k: usize) {
    if xs.len() <= k {
        xs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        return;
    }
    // Partial-sort via select_nth-like: simple since we have std only.
    xs.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
    xs[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
}

// ---------------------------------------------------------------------------
// Searcher trait so backends are swappable.
// ---------------------------------------------------------------------------

pub trait Searcher {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>;
    fn name(&self) -> &'static str;
}

impl Searcher for BaselinePq {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        BaselinePq::search(self, query, k)
    }
    fn name(&self) -> &'static str {
        "pq_baseline"
    }
}

impl Searcher for DominantSkipPq {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        DominantSkipPq::search(self, query, k)
    }
    fn name(&self) -> &'static str {
        "pq_dominant_skip"
    }
}

// ---------------------------------------------------------------------------
// Synthetic clustered data generator. Clustered data produces realistic
// non-uniform PQ code distributions (hubness in subspaces).
// ---------------------------------------------------------------------------

pub fn clustered_dataset(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    // Generate cluster centers.
    let mut centers = vec![0.0f32; n_clusters * dim];
    for v in centers.iter_mut() {
        *v = rng.gauss() * 2.5;
    }
    let mut data = Dataset::new(n, dim);
    for i in 0..n {
        let c = rng.range(n_clusters);
        let row = data.row_mut(i);
        for d in 0..dim {
            row[d] = centers[c * dim + d] + rng.gauss() * 0.4;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Benchmark helper (used by both examples/bench.rs and tests).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BenchRow {
    pub name: &'static str,
    pub per_query_us: f64,
    pub recall: f64,
    pub bytes: u64,
}

pub fn recall_at_k(gt: &[(u32, f32)], got: &[(u32, f32)]) -> f64 {
    let g: std::collections::HashSet<u32> = gt.iter().map(|x| x.0).collect();
    let hits = got.iter().filter(|x| g.contains(&x.0)).count();
    hits as f64 / gt.len() as f64
}

pub fn run_search<S: Searcher>(
    s: &S,
    queries: &Dataset,
    truth: &[Vec<(u32, f32)>],
    k: usize,
) -> BenchRow {
    let t0 = Instant::now();
    let mut recall_sum = 0.0;
    for qi in 0..queries.n {
        let r = s.search(queries.row(qi), k);
        recall_sum += recall_at_k(&truth[qi], &r);
    }
    let elapsed = t0.elapsed();
    BenchRow {
        name: s.name(),
        per_query_us: elapsed.as_secs_f64() * 1e6 / queries.n as f64,
        recall: recall_sum / queries.n as f64,
        bytes: 0,
    }
}
