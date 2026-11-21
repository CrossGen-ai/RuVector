//! Polysemous Product Quantization with Hamming Prefilter + ADC Rerank.
//!
//! Based on Douze, Jégou & Schmid, "Polysemous codes" (ECCV 2016).
//! Standard PQ codes are reordered per-subquantizer so that Hamming distance
//! between codes correlates with L2 distance between the original centroids.
//! Query time uses a two-stage filter:
//!   1. Cheap Hamming distance between query code and database codes, filter
//!      out items whose Hamming distance exceeds a threshold.
//!   2. ADC (asymmetric distance computation) on survivors only.
//!
//! This crate is intentionally dependency-light (rand + thiserror) and runs
//! entirely in safe Rust. All benchmark numbers are produced by the bundled
//! `polysemous-bench` binary against synthetic data — no mocks.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PqError {
    #[error("invalid dimension: dim {dim} must be divisible by m {m}")]
    InvalidDim { dim: usize, m: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("ksub must be 16 or 256 for byte-packed codes, got {0}")]
    BadKsub(usize),
}

/// A trained Product Quantizer. `m` subspaces, `ksub` centroids each.
/// Codes are stored as one byte per subspace (works for ksub up to 256).
#[derive(Clone, Debug)]
pub struct Pq {
    pub dim: usize,
    pub m: usize,
    pub dsub: usize,
    pub ksub: usize,
    /// codebooks[s]: ksub rows of dsub f32 — row-major.
    pub codebooks: Vec<Vec<f32>>,
    /// perm[s][old_idx] = new_idx (polysemous reorder). Identity by default.
    pub perm: Vec<Vec<u8>>,
    /// inv_perm[s][new_idx] = old_idx.
    pub inv_perm: Vec<Vec<u8>>,
}

impl Pq {
    /// Train `m` k-means codebooks of size `ksub` over `train` (n×dim row-major).
    pub fn train(
        train: &[f32],
        dim: usize,
        m: usize,
        ksub: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, PqError> {
        if dim % m != 0 {
            return Err(PqError::InvalidDim { dim, m });
        }
        if !(ksub == 16 || ksub == 256) {
            return Err(PqError::BadKsub(ksub));
        }
        let n = train.len() / dim;
        if n == 0 {
            return Err(PqError::EmptyTraining);
        }
        let dsub = dim / m;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut codebooks = Vec::with_capacity(m);
        for s in 0..m {
            let mut subdata = vec![0f32; n * dsub];
            for i in 0..n {
                let src = &train[i * dim + s * dsub..i * dim + (s + 1) * dsub];
                subdata[i * dsub..(i + 1) * dsub].copy_from_slice(src);
            }
            let cb = kmeans(&subdata, dsub, ksub, iters, &mut rng);
            codebooks.push(cb);
        }
        let perm: Vec<Vec<u8>> = (0..m)
            .map(|_| (0..ksub as u16).map(|i| i as u8).collect())
            .collect();
        let inv_perm = perm.clone();
        Ok(Pq {
            dim,
            m,
            dsub,
            ksub,
            codebooks,
            perm,
            inv_perm,
        })
    }

    /// Encode a single vector to `m` bytes (no polysemous remap).
    pub fn encode_raw(&self, v: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.m];
        for s in 0..self.m {
            let sub = &v[s * self.dsub..(s + 1) * self.dsub];
            let mut best = 0u16;
            let mut best_d = f32::INFINITY;
            for k in 0..self.ksub as u16 {
                let c = &self.codebooks[s][k as usize * self.dsub..(k as usize + 1) * self.dsub];
                let d = l2_sq(sub, c);
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            code[s] = best as u8;
        }
        code
    }

    /// Encode with polysemous remap applied.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let raw = self.encode_raw(v);
        let mut out = vec![0u8; self.m];
        for s in 0..self.m {
            out[s] = self.perm[s][raw[s] as usize];
        }
        out
    }

    /// Build ADC lookup table: m × ksub f32 of partial L2 squared, indexed by *remapped* code byte.
    /// So `lut[s * ksub + perm[s][old_k]]` holds L2sq(q_sub, centroid_old_k).
    pub fn adc_lut(&self, q: &[f32]) -> Vec<f32> {
        let mut lut = vec![0f32; self.m * self.ksub];
        for s in 0..self.m {
            let qs = &q[s * self.dsub..(s + 1) * self.dsub];
            for k in 0..self.ksub {
                let c = &self.codebooks[s][k * self.dsub..(k + 1) * self.dsub];
                let d = l2_sq(qs, c);
                let remapped = self.perm[s][k] as usize;
                lut[s * self.ksub + remapped] = d;
            }
        }
        lut
    }

    /// Sum ADC table over a code to get approximate L2sq.
    pub fn adc_distance(&self, lut: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0f32;
        for s in 0..self.m {
            sum += lut[s * self.ksub + code[s] as usize];
        }
        sum
    }

    /// Fit a polysemous reordering by simulated annealing per-subquantizer.
    /// Objective per subspace s: maximize correlation between Hamming distance
    /// (between remapped code bytes) and L2 distance between centroids.
    /// We minimize cost = Σ_{i<j} (hamming(π(i), π(j)) - α · d(c_i, c_j))² with α auto-fit.
    pub fn fit_polysemous(&mut self, sa_iters: usize, seed: u64) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let bits = if self.ksub == 256 { 8 } else { 4 };
        for s in 0..self.m {
            let cb = &self.codebooks[s];
            // Precompute pairwise L2 and Hamming for this subspace.
            let k = self.ksub;
            let mut l2 = vec![0f32; k * k];
            for i in 0..k {
                for j in i + 1..k {
                    let a = &cb[i * self.dsub..(i + 1) * self.dsub];
                    let b = &cb[j * self.dsub..(j + 1) * self.dsub];
                    let d = l2_sq(a, b).sqrt();
                    l2[i * k + j] = d;
                    l2[j * k + i] = d;
                }
            }
            // Hamming table over bit codes 0..k.
            let mut ham = vec![0u8; k * k];
            for i in 0..k {
                for j in 0..k {
                    ham[i * k + j] = ((i ^ j) as u32).count_ones() as u8;
                }
            }
            // Fit scalar α = (Σ h·d) / (Σ d²), minimizing cost.
            let mut perm: Vec<u8> = (0..k as u16).map(|x| x as u8).collect();
            let mut cost = eval_cost(&perm, &l2, &ham, k);
            let t0 = cost.max(1e-6) / (k as f32);
            let iters = sa_iters.max(1000);
            for step in 0..iters {
                let t = t0 * (1.0 - step as f32 / iters as f32).max(1e-3);
                let i = rng.gen_range(0..k);
                let mut j = rng.gen_range(0..k);
                while j == i {
                    j = rng.gen_range(0..k);
                }
                perm.swap(i, j);
                let new_cost = eval_cost(&perm, &l2, &ham, k);
                let d = new_cost - cost;
                if d < 0.0 || rng.gen::<f32>() < (-d / t).exp() {
                    cost = new_cost;
                } else {
                    perm.swap(i, j);
                }
            }
            // Build inverse permutation.
            let mut inv = vec![0u8; k];
            for (old, &new) in perm.iter().enumerate() {
                inv[new as usize] = old as u8;
            }
            self.perm[s] = perm;
            self.inv_perm[s] = inv;
            let _ = bits; // bit count kept for documentation / asserts.
        }
    }
}

fn eval_cost(perm: &[u8], l2: &[f32], ham: &[u8], k: usize) -> f32 {
    // cost = Σ_{i<j} (hamming(perm[i],perm[j]) - α·d(i,j))²
    // α fit: α = Σ h·d / Σ d²
    let mut sum_hd = 0f32;
    let mut sum_dd = 0f32;
    for i in 0..k {
        for j in i + 1..k {
            let d = l2[i * k + j];
            let h = ham[perm[i] as usize * k + perm[j] as usize] as f32;
            sum_hd += h * d;
            sum_dd += d * d;
        }
    }
    let alpha = if sum_dd > 1e-12 { sum_hd / sum_dd } else { 0.0 };
    let mut c = 0f32;
    for i in 0..k {
        for j in i + 1..k {
            let d = l2[i * k + j];
            let h = ham[perm[i] as usize * k + perm[j] as usize] as f32;
            let e = h - alpha * d;
            c += e * e;
        }
    }
    c
}

fn kmeans(data: &[f32], dsub: usize, k: usize, iters: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let n = data.len() / dsub;
    assert!(n >= k, "need at least k points to seed kmeans (n={n}, k={k})");
    // Init: random distinct points.
    let mut used = vec![false; n];
    let mut centers = vec![0f32; k * dsub];
    for ki in 0..k {
        loop {
            let idx = rng.gen_range(0..n);
            if !used[idx] {
                used[idx] = true;
                centers[ki * dsub..(ki + 1) * dsub]
                    .copy_from_slice(&data[idx * dsub..(idx + 1) * dsub]);
                break;
            }
        }
    }
    let mut assign = vec![0u16; n];
    for _ in 0..iters {
        // Assign.
        for i in 0..n {
            let x = &data[i * dsub..(i + 1) * dsub];
            let mut best = 0u16;
            let mut best_d = f32::INFINITY;
            for ki in 0..k {
                let c = &centers[ki * dsub..(ki + 1) * dsub];
                let d = l2_sq(x, c);
                if d < best_d {
                    best_d = d;
                    best = ki as u16;
                }
            }
            assign[i] = best;
        }
        // Update.
        let mut sums = vec![0f32; k * dsub];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let a = assign[i] as usize;
            counts[a] += 1;
            let x = &data[i * dsub..(i + 1) * dsub];
            for d in 0..dsub {
                sums[a * dsub + d] += x[d];
            }
        }
        for ki in 0..k {
            if counts[ki] == 0 {
                // Reinit empty cluster with random point.
                let idx = rng.gen_range(0..n);
                centers[ki * dsub..(ki + 1) * dsub]
                    .copy_from_slice(&data[idx * dsub..(idx + 1) * dsub]);
            } else {
                let c = counts[ki] as f32;
                for d in 0..dsub {
                    centers[ki * dsub + d] = sums[ki * dsub + d] / c;
                }
            }
        }
    }
    centers
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Pack codes column-major into one buffer: `codes[i*m + s]` = subspace s byte of vector i.
pub fn pack_codes(pq: &Pq, vectors: &[f32]) -> Vec<u8> {
    let n = vectors.len() / pq.dim;
    let mut out = vec![0u8; n * pq.m];
    for i in 0..n {
        let code = pq.encode(&vectors[i * pq.dim..(i + 1) * pq.dim]);
        out[i * pq.m..(i + 1) * pq.m].copy_from_slice(&code);
    }
    out
}

#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    let mut s = 0u32;
    for i in 0..a.len() {
        s += (a[i] ^ b[i]).count_ones();
    }
    s
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub idx: u32,
    pub dist: f32,
}

/// Polysemous search: Hamming prefilter → ADC rerank.
/// Returns top-k results ordered by ADC distance.
pub fn polysemous_search(
    pq: &Pq,
    codes: &[u8],
    n: usize,
    query: &[f32],
    ham_threshold: u32,
    top_k: usize,
) -> (Vec<Candidate>, usize) {
    let q_code = pq.encode(query);
    let lut = pq.adc_lut(query);
    let m = pq.m;
    let mut heap: Vec<Candidate> = Vec::with_capacity(top_k + 1);
    let mut survivors = 0usize;
    for i in 0..n {
        let c = &codes[i * m..(i + 1) * m];
        if hamming(&q_code, c) > ham_threshold {
            continue;
        }
        survivors += 1;
        let d = pq.adc_distance(&lut, c);
        push_topk(&mut heap, Candidate { idx: i as u32, dist: d }, top_k);
    }
    heap.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    (heap, survivors)
}

/// ADC-only search (no prefilter) for comparison.
pub fn adc_search(pq: &Pq, codes: &[u8], n: usize, query: &[f32], top_k: usize) -> Vec<Candidate> {
    let lut = pq.adc_lut(query);
    let m = pq.m;
    let mut heap: Vec<Candidate> = Vec::with_capacity(top_k + 1);
    for i in 0..n {
        let c = &codes[i * m..(i + 1) * m];
        let d = pq.adc_distance(&lut, c);
        push_topk(&mut heap, Candidate { idx: i as u32, dist: d }, top_k);
    }
    heap.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    heap
}

/// Full-scan exact L2sq for ground truth.
pub fn exact_search(vectors: &[f32], dim: usize, n: usize, query: &[f32], top_k: usize) -> Vec<Candidate> {
    let mut heap: Vec<Candidate> = Vec::with_capacity(top_k + 1);
    for i in 0..n {
        let d = l2_sq(&vectors[i * dim..(i + 1) * dim], query);
        push_topk(&mut heap, Candidate { idx: i as u32, dist: d }, top_k);
    }
    heap.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    heap
}

fn push_topk(heap: &mut Vec<Candidate>, c: Candidate, k: usize) {
    if heap.len() < k {
        heap.push(c);
        return;
    }
    // find max
    let mut max_i = 0usize;
    for i in 1..heap.len() {
        if heap[i].dist > heap[max_i].dist {
            max_i = i;
        }
    }
    if c.dist < heap[max_i].dist {
        heap[max_i] = c;
    }
}

/// Recall@k: fraction of ground_truth indices present in `candidates` top-k.
pub fn recall(ground_truth: &[Candidate], candidates: &[Candidate], k: usize) -> f32 {
    let gt: std::collections::HashSet<u32> =
        ground_truth.iter().take(k).map(|c| c.idx).collect();
    let hits = candidates
        .iter()
        .take(k)
        .filter(|c| gt.contains(&c.idx))
        .count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn train_encode_roundtrip() {
        let data = make_data(1024, 32, 1);
        let pq = Pq::train(&data, 32, 4, 16, 10, 7).unwrap();
        let code = pq.encode(&data[0..32]);
        assert_eq!(code.len(), 4);
        for &b in &code {
            assert!((b as usize) < 16);
        }
    }

    #[test]
    fn polysemous_reduces_cost() {
        let data = make_data(2048, 32, 2);
        let mut pq = Pq::train(&data, 32, 4, 16, 15, 3).unwrap();
        // measure cost before
        let k = pq.ksub;
        let mut l2 = vec![0f32; k * k];
        let mut ham = vec![0u8; k * k];
        let cb = &pq.codebooks[0];
        for i in 0..k {
            for j in 0..k {
                let a = &cb[i * pq.dsub..(i + 1) * pq.dsub];
                let b = &cb[j * pq.dsub..(j + 1) * pq.dsub];
                l2[i * k + j] = l2_sq(a, b).sqrt();
                ham[i * k + j] = ((i ^ j) as u32).count_ones() as u8;
            }
        }
        let identity: Vec<u8> = (0..k as u16).map(|x| x as u8).collect();
        let c_before = eval_cost(&identity, &l2, &ham, k);
        pq.fit_polysemous(5000, 11);
        let c_after = eval_cost(&pq.perm[0], &l2, &ham, k);
        assert!(c_after < c_before, "polysemous SA should reduce cost: {c_before} -> {c_after}");
    }

    #[test]
    fn search_returns_k_results() {
        let n = 512;
        let dim = 16;
        let data = make_data(n, dim, 4);
        let mut pq = Pq::train(&data, dim, 4, 16, 10, 5).unwrap();
        pq.fit_polysemous(2000, 6);
        let codes = pack_codes(&pq, &data);
        let (res, _) = polysemous_search(&pq, &codes, n, &data[0..dim], 16, 10);
        assert_eq!(res.len(), 10);
    }
}
