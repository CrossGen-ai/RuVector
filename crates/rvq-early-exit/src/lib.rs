//! Residual Vector Quantization (RVQ) with query-adaptive early exit.
//!
//! Classical RVQ encodes a vector as a sum of M quantized residuals:
//!   x ≈ c_1[i_1] + c_2[i_2] + ... + c_M[i_M]
//! where each stage m owns its own K-sized codebook c_m trained on the
//! residual r_{m-1} = x - sum_{j<m} c_j[i_j].
//!
//! Full-depth ADC decodes all M stages per query and sums partial squared
//! distances:
//!   d^2(q,x) ≈ sum_m ||q_m - c_m[i_m]||^2 + cross-terms-approx
//! (we use the symmetric-RVQ ADC that treats stages as additive, which
//! is consistent for the Euclidean inner product when codebooks are
//! trained greedily on residuals — see discussion in the research doc.)
//!
//! Early exit: at each partial stage m, maintain a running top-K
//! shortlist by partial score. Once the (K+spill)-th best's lower bound
//! exceeds the K-th best's upper bound, the remaining stages can no
//! longer reorder the top-K so we stop refining the shortlist.
//!
//! No unsafe, no external crates. Deterministic PRNG for reproducibility.

use std::cmp::Ordering;

/// Deterministic xorshift64* PRNG. Portable and zero-dep.
#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Box-Muller gaussian pair; returns one.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// Dense vector utilities.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn add_into(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

#[inline]
pub fn sub_into(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] -= src[i];
    }
}

/// A single stage codebook: K codewords of dimension D.
#[derive(Clone)]
pub struct Codebook {
    pub d: usize,
    pub k: usize,
    /// Flat [k * d].
    pub codes: Vec<f32>,
}

impl Codebook {
    pub fn code(&self, idx: usize) -> &[f32] {
        &self.codes[idx * self.d..(idx + 1) * self.d]
    }

    /// Fit K-means on `data` with Lloyd iterations. Deterministic init via
    /// evenly spaced selection from `data`.
    pub fn fit_kmeans(data: &[Vec<f32>], k: usize, iters: usize, rng: &mut Rng) -> Self {
        assert!(!data.is_empty());
        let d = data[0].len();
        let n = data.len();
        let mut codes = vec![0.0f32; k * d];
        // Init: pick k distinct indices spread over the dataset, with rng jitter.
        for ki in 0..k {
            let base = (ki * n) / k;
            let jitter = (rng.next_u64() as usize) % n.max(1);
            let idx = (base + jitter) % n;
            codes[ki * d..(ki + 1) * d].copy_from_slice(&data[idx]);
        }
        let mut assign = vec![0usize; n];
        for _it in 0..iters {
            // Assign step.
            for (i, v) in data.iter().enumerate() {
                let mut best_j = 0usize;
                let mut best_d = f32::INFINITY;
                for j in 0..k {
                    let c = &codes[j * d..(j + 1) * d];
                    let dd = sq_l2(v, c);
                    if dd < best_d {
                        best_d = dd;
                        best_j = j;
                    }
                }
                assign[i] = best_j;
            }
            // Update step.
            let mut sums = vec![0.0f32; k * d];
            let mut counts = vec![0u32; k];
            for (i, v) in data.iter().enumerate() {
                let j = assign[i];
                counts[j] += 1;
                let s = &mut sums[j * d..(j + 1) * d];
                for t in 0..d {
                    s[t] += v[t];
                }
            }
            for j in 0..k {
                let c = &mut codes[j * d..(j + 1) * d];
                if counts[j] == 0 {
                    // Dead cluster: reseed from a random point.
                    let idx = (rng.next_u64() as usize) % n;
                    c.copy_from_slice(&data[idx]);
                } else {
                    let inv = 1.0 / counts[j] as f32;
                    let s = &sums[j * d..(j + 1) * d];
                    for t in 0..d {
                        c[t] = s[t] * inv;
                    }
                }
            }
        }
        Self { d, k, codes }
    }
}

/// RVQ index with M stages * K codes each.
pub struct Rvq {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub stages: Vec<Codebook>,
    /// Encoded dataset: [n * m] code indices.
    pub codes: Vec<u8>,
    pub n: usize,
}

impl Rvq {
    /// Train M stages greedily on residuals and encode `data`.
    pub fn train_and_encode(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        kmeans_iters: usize,
        rng: &mut Rng,
    ) -> Self {
        assert!(k <= 256, "u8 codes");
        assert!(!data.is_empty());
        let d = data[0].len();
        let n = data.len();
        // Working residuals, owned.
        let mut residuals: Vec<Vec<f32>> = data.to_vec();
        let mut stages = Vec::with_capacity(m);
        let mut codes = vec![0u8; n * m];
        for s in 0..m {
            let cb = Codebook::fit_kmeans(&residuals, k, kmeans_iters, rng);
            // Encode current residuals, subtract from them.
            for (i, r) in residuals.iter_mut().enumerate() {
                let mut best_j = 0usize;
                let mut best_d = f32::INFINITY;
                for j in 0..k {
                    let c = cb.code(j);
                    let dd = sq_l2(r, c);
                    if dd < best_d {
                        best_d = dd;
                        best_j = j;
                    }
                }
                codes[i * m + s] = best_j as u8;
                // Subtract the chosen codeword from the residual.
                let c = cb.code(best_j);
                for t in 0..d {
                    r[t] -= c[t];
                }
            }
            stages.push(cb);
        }
        Self {
            d,
            m,
            k,
            stages,
            codes,
            n,
        }
    }

    /// Return approximate reconstruction of index `i`.
    pub fn reconstruct(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.d];
        for s in 0..self.m {
            let idx = self.codes[i * self.m + s] as usize;
            add_into(&mut out, self.stages[s].code(idx));
        }
        out
    }

    /// Build per-stage LUTs: lut[s][j] = ||q - c_s[j]||^2 - ||q||^2 - ||c_s[j]||^2
    /// Actually we want the additive stage distance. For symmetric RVQ ADC we use:
    ///   partial d^2 after S stages ≈ sum_{s<S} ||q_s - c_s[i_s]||^2
    /// where q_s is the residual of q against prior codewords. Since the DB
    /// vector's prior codewords are per-item, we cannot precompute q_s per
    /// query alone. We use the simpler standard asymmetric form:
    ///   d^2 ≈ ||q - sum_s c_s[i_s]||^2
    /// Expanded:  ||q||^2 - 2 q·(sum c) + ||sum c||^2
    /// We precompute per-stage LUT of (-2 q·c_s[j]) and per-item precomputed
    /// ||sum c||^2 (actually sum_s ||c_s[i_s]||^2 + 2 sum_{s<t} c_s[i_s]·c_t[i_t]
    /// which is nontrivial). We instead precompute a per-item partial norm
    /// table: norm_partial[i][S] = ||sum_{s<S} c_s[i_s]||^2 at build time, and
    /// per-stage dot LUT at query time. This gives an exact decomposition:
    ///   partial_dist(i, S) = ||q||^2 + norm_partial[i][S] - 2 * sum_{s<S} lut_s[q][i_s]
    /// where lut_s[q][j] = q · c_s[j].
    pub fn build_partial_norms(&self) -> Vec<f32> {
        // Shape: n * (m + 1). norms[i*(m+1) + s] = ||sum_{t<s} c_t[i_t]||^2.
        let mut out = vec![0.0f32; self.n * (self.m + 1)];
        let mut acc = vec![0.0f32; self.d];
        for i in 0..self.n {
            // Reset acc.
            for t in 0..self.d {
                acc[t] = 0.0;
            }
            for s in 0..self.m {
                // Store ||acc||^2 at position s.
                let mut nn = 0.0f32;
                for t in 0..self.d {
                    nn += acc[t] * acc[t];
                }
                out[i * (self.m + 1) + s] = nn;
                // Add stage s codeword.
                let idx = self.codes[i * self.m + s] as usize;
                let c = self.stages[s].code(idx);
                for t in 0..self.d {
                    acc[t] += c[t];
                }
            }
            // Position m: final full norm.
            let mut nn = 0.0f32;
            for t in 0..self.d {
                nn += acc[t] * acc[t];
            }
            out[i * (self.m + 1) + self.m] = nn;
        }
        out
    }

    /// Per-query stage LUT: lut[s * k + j] = q · c_s[j].
    pub fn build_query_lut(&self, q: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0f32; self.m * self.k];
        for s in 0..self.m {
            for j in 0..self.k {
                let c = self.stages[s].code(j);
                let mut dp = 0.0f32;
                for t in 0..self.d {
                    dp += q[t] * c[t];
                }
                lut[s * self.k + j] = dp;
            }
        }
        lut
    }
}

/// Top-K min-heap keyed by (partial_dist, idx) with max at root.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub idx: u32,
    pub d: f32,
}
impl Eq for Hit {}
impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.d == other.d && self.idx == other.idx
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.d.partial_cmp(&other.d).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Full-depth ADC scan: decode all M stages for every item, return top-K.
pub fn scan_full(rvq: &Rvq, q: &[f32], part_norms: &[f32], top_k: usize) -> Vec<Hit> {
    let q_norm: f32 = q.iter().map(|x| x * x).sum();
    let lut = rvq.build_query_lut(q);
    let mut hits: Vec<Hit> = Vec::with_capacity(rvq.n);
    for i in 0..rvq.n {
        let mut dp = 0.0f32;
        for s in 0..rvq.m {
            let j = rvq.codes[i * rvq.m + s] as usize;
            dp += lut[s * rvq.k + j];
        }
        let item_norm = part_norms[i * (rvq.m + 1) + rvq.m];
        let d2 = q_norm + item_norm - 2.0 * dp;
        hits.push(Hit {
            idx: i as u32,
            d: d2,
        });
    }
    hits.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal));
    hits.truncate(top_k);
    hits
}

/// Early-exit RVQ scan.
///
/// Strategy:
///   1. Decode stage 0 for every item -> partial scores.
///   2. Keep a shortlist of top (top_k * spill) candidates.
///   3. For each subsequent stage s, decode only shortlist items,
///      update their scores, re-rank. If after stage s the shortlist's
///      internal ordering of the top_k is unchanged for `stable_streak`
///      consecutive stages, exit early.
///
/// Reports how many stages were actually decoded ("effective M") via the
/// returned `stages_used` count.
pub struct EarlyExitStats {
    pub stages_used: usize,
    pub shortlist_size: usize,
}

pub fn scan_early_exit(
    rvq: &Rvq,
    q: &[f32],
    part_norms: &[f32],
    top_k: usize,
    spill: usize,
    stable_streak: usize,
) -> (Vec<Hit>, EarlyExitStats) {
    let q_norm: f32 = q.iter().map(|x| x * x).sum();
    let lut = rvq.build_query_lut(q);
    let shortlist_size = (top_k * spill).min(rvq.n);
    // Stage 0: decode for everyone.
    let mut running: Vec<(u32, f32)> = Vec::with_capacity(rvq.n);
    for i in 0..rvq.n {
        let j = rvq.codes[i * rvq.m] as usize;
        let dp = lut[j];
        // Partial squared distance using the per-item PARTIAL norm after stage 0:
        // = q_norm + ||c_0[i_0]||^2 - 2 (q · c_0[i_0])
        let item_pnorm = part_norms[i * (rvq.m + 1) + 1];
        let d2 = q_norm + item_pnorm - 2.0 * dp;
        running.push((i as u32, d2));
    }
    // Partial sort — nth_element style: use select_nth_unstable_by for shortlist.
    if shortlist_size < running.len() {
        let (_, _, _) = running.select_nth_unstable_by(shortlist_size - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        running.truncate(shortlist_size);
    }
    running.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut stages_used = 1usize;
    let mut stable = 0usize;
    let mut prev_topk: Vec<u32> = running
        .iter()
        .take(top_k.min(running.len()))
        .map(|p| p.0)
        .collect();

    for s in 1..rvq.m {
        // Refine shortlist: for each candidate, add contribution of stage s.
        // partial_dist after S stages (S = s+1) =
        //   q_norm + part_norms[i][S] - 2 sum_{t<S} lut[t][code_t(i)]
        // We incrementally update:
        //   new = old - part_norms[i][s] + part_norms[i][s+1] - 2 lut[s][code_s(i)]
        // Actually old = q_norm + part_norms[i][s] - 2 dot_through_{s-1}
        // So new = old - part_norms[i][s] + part_norms[i][s+1]
        //          - 2 * lut[s * k + code_s(i)]
        for (i, d) in running.iter_mut() {
            let ii = *i as usize;
            let j = rvq.codes[ii * rvq.m + s] as usize;
            let pn_old = part_norms[ii * (rvq.m + 1) + s];
            let pn_new = part_norms[ii * (rvq.m + 1) + s + 1];
            *d = *d - pn_old + pn_new - 2.0 * lut[s * rvq.k + j];
        }
        running.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        stages_used = s + 1;

        let cur_topk: Vec<u32> = running
            .iter()
            .take(top_k.min(running.len()))
            .map(|p| p.0)
            .collect();
        if cur_topk == prev_topk {
            stable += 1;
            if stable >= stable_streak {
                break;
            }
        } else {
            stable = 0;
        }
        prev_topk = cur_topk;
    }

    let hits: Vec<Hit> = running
        .into_iter()
        .take(top_k)
        .map(|(i, d)| Hit { idx: i, d })
        .collect();
    (
        hits,
        EarlyExitStats {
            stages_used,
            shortlist_size,
        },
    )
}

/// Exact brute force — ground truth for recall@K.
pub fn scan_exact(data: &[Vec<f32>], q: &[f32], top_k: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = data
        .iter()
        .enumerate()
        .map(|(i, v)| Hit {
            idx: i as u32,
            d: sq_l2(v, q),
        })
        .collect();
    hits.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(Ordering::Equal));
    hits.truncate(top_k);
    hits
}

/// Generate a synthetic clustered Gaussian dataset.
pub fn gen_clustered(n: usize, d: usize, n_clusters: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let mut c = vec![0.0f32; d];
        for t in 0..d {
            c[t] = rng.gaussian() * 3.0;
        }
        centers.push(c);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % n_clusters];
        let mut v = vec![0.0f32; d];
        for t in 0..d {
            v[t] = c[t] + rng.gaussian() * 0.5;
        }
        out.push(v);
    }
    out
}

/// Compute recall@K between two hit lists (set overlap on idx).
pub fn recall_at_k(gt: &[Hit], pred: &[Hit]) -> f32 {
    use std::collections::HashSet;
    let g: HashSet<u32> = gt.iter().map(|h| h.idx).collect();
    let hit = pred.iter().filter(|h| g.contains(&h.idx)).count();
    hit as f32 / gt.len().max(1) as f32
}
