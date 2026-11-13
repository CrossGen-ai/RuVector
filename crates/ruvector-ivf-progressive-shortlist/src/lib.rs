//! IVF-PQ with Progressive Shortlist (IPSL).
//!
//! Baseline IVF-PQ scans every code in each probed posting list.
//! IPSL does breadth-first scanning: visit lists in order of
//! query-to-centroid score; within each list, scan only `m_init`
//! codes in a precomputed residual-norm-desc order; then re-enter
//! lists whose remaining tail could still improve the global top-k
//! (bounded by `||q|| * max_residual_norm_in_tail`).
//!
//! Scanners share a single Scanner trait so backends stay swappable.

use std::cmp::Ordering;

pub type Scalar = f32;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: u32,
    pub score: Scalar, // larger == better (negative L2 or dot)
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        // min-heap on score (BinaryHeap is max; we flip in heap usage)
        self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal)
    }
}

/// Simple top-k via sorted insertion (small k keeps this competitive with a
/// binary heap and avoids BinaryHeap<Candidate> flip bookkeeping).
pub struct TopK {
    k: usize,
    pub items: Vec<Candidate>,
}
impl TopK {
    pub fn new(k: usize) -> Self { Self { k, items: Vec::with_capacity(k + 1) } }
    pub fn threshold(&self) -> Scalar {
        if self.items.len() < self.k { Scalar::NEG_INFINITY }
        else { self.items.last().unwrap().score }
    }
    pub fn push(&mut self, c: Candidate) {
        if self.items.len() >= self.k && c.score <= self.items.last().unwrap().score {
            return;
        }
        // insert sorted desc by score
        let pos = self.items.binary_search_by(|a| {
            c.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        }).unwrap_or_else(|p| p);
        self.items.insert(pos, c);
        if self.items.len() > self.k { self.items.pop(); }
    }
    pub fn into_sorted(self) -> Vec<Candidate> { self.items }
}

/// A single IVF posting list.
#[derive(Clone, Debug)]
pub struct Posting {
    pub centroid: Vec<Scalar>,
    pub ids: Vec<u32>,
    /// Residual (vec - centroid) for each id. Kept uncompressed for a clean
    /// apples-to-apples benchmark; a PQ backend swaps this for codes + LUT.
    pub residuals: Vec<Vec<Scalar>>,
    /// Residual norms descending-sorted; `order[i]` is the position of the
    /// i-th-largest-residual-norm element in `ids`/`residuals`.
    pub order: Vec<u32>,
    /// `max_norm_tail[i]` = max residual norm among the elements NOT yet
    /// scanned after visiting the first i in `order`. Equals
    /// `norms[order[i]]` because `order` is desc-sorted.
    pub max_norm_tail: Vec<Scalar>,
}

impl Posting {
    pub fn build(centroid: Vec<Scalar>, ids: Vec<u32>, vectors: &[Vec<Scalar>]) -> Self {
        assert_eq!(ids.len(), vectors.len());
        let dim = centroid.len();
        let residuals: Vec<Vec<Scalar>> = vectors.iter().map(|v| {
            let mut r = vec![0.0; dim];
            for d in 0..dim { r[d] = v[d] - centroid[d]; }
            r
        }).collect();
        let mut idx_norm: Vec<(u32, Scalar)> = residuals.iter().enumerate()
            .map(|(i, r)| (i as u32, dot(r, r).sqrt()))
            .collect();
        idx_norm.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let order: Vec<u32> = idx_norm.iter().map(|x| x.0).collect();
        // max_norm_tail[i] = norm at position i in the sorted order (the max
        // among the not-yet-scanned IF we scan prefix 0..i).
        let max_norm_tail: Vec<Scalar> = idx_norm.iter().map(|x| x.1).collect();
        Self { centroid, ids, residuals, order, max_norm_tail }
    }
}

#[inline]
pub fn dot(a: &[Scalar], b: &[Scalar]) -> Scalar {
    let mut s = 0.0;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}
#[inline]
pub fn norm(a: &[Scalar]) -> Scalar { dot(a, a).sqrt() }

/// Score for (query, vec) = negative L2^2.  All backends must agree.
#[inline]
pub fn neg_l2sq(q: &[Scalar], v: &[Scalar]) -> Scalar {
    let mut s = 0.0;
    for i in 0..q.len() { let d = q[i] - v[i]; s -= d * d; }
    s
}

/// Score for (query, centroid+residual). Avoids re-adding the centroid: we
/// reconstruct score from centroid score plus residual correction.
/// `-||q - (c + r)||^2 = -||q-c||^2 + 2 (q-c)·r - ||r||^2`
#[inline]
pub fn score_from_residual(q_minus_c: &[Scalar], qc_neg_l2sq: Scalar, r: &[Scalar]) -> Scalar {
    let mut dot_qcr = 0.0;
    let mut norm_r2 = 0.0;
    for i in 0..r.len() {
        dot_qcr += q_minus_c[i] * r[i];
        norm_r2 += r[i] * r[i];
    }
    qc_neg_l2sq + 2.0 * dot_qcr - norm_r2
}

/// Scanner trait — backends plug in here.
pub trait Scanner {
    fn name(&self) -> &'static str;
    fn search(&self, postings: &[Posting], query: &[Scalar], nprobe: usize, k: usize) -> Vec<Candidate>;
}

/// B0 — exhaustive IVF: scan every residual in every probed list, in
/// whatever order they appear.
pub struct Exhaustive;
impl Scanner for Exhaustive {
    fn name(&self) -> &'static str { "exhaustive" }
    fn search(&self, postings: &[Posting], query: &[Scalar], nprobe: usize, k: usize) -> Vec<Candidate> {
        let probes = pick_probes(postings, query, nprobe);
        let mut topk = TopK::new(k);
        for &pi in &probes {
            let p = &postings[pi];
            let q_minus_c = sub(query, &p.centroid);
            let qc = -dot(&q_minus_c, &q_minus_c);
            for i in 0..p.ids.len() {
                let s = score_from_residual(&q_minus_c, qc, &p.residuals[i]);
                topk.push(Candidate { id: p.ids[i], score: s });
            }
        }
        topk.into_sorted()
    }
}

/// B1 — partial-norm-ordered: scan ALL residuals in each list but in
/// descending residual-norm order and stop within a list once the next
/// element's upper bound `qc + 2*||q-c||*||r|| - ||r||^2` falls below the
/// current top-k threshold (Cauchy–Schwarz).
pub struct NormOrderedAbandon;
impl Scanner for NormOrderedAbandon {
    fn name(&self) -> &'static str { "norm-ordered-abandon" }
    fn search(&self, postings: &[Posting], query: &[Scalar], nprobe: usize, k: usize) -> Vec<Candidate> {
        let probes = pick_probes(postings, query, nprobe);
        let mut topk = TopK::new(k);
        for &pi in &probes {
            let p = &postings[pi];
            let q_minus_c = sub(query, &p.centroid);
            let qcn = norm(&q_minus_c);
            let qc = -(qcn * qcn);
            for i in 0..p.order.len() {
                let pos = p.order[i] as usize;
                let rn = p.max_norm_tail[i];
                // Upper bound on any residual with norm ≤ rn.
                let ub = qc + 2.0 * qcn * rn - rn * rn;
                if ub < topk.threshold() { break; }
                let s = score_from_residual(&q_minus_c, qc, &p.residuals[pos]);
                topk.push(Candidate { id: p.ids[pos], score: s });
            }
        }
        topk.into_sorted()
    }
}

/// B2 — progressive breadth-first: scan `m_init` codes per probed list,
/// then revisit lists whose remaining tail could still improve top-k.
pub struct Progressive { pub m_init: usize }
impl Scanner for Progressive {
    fn name(&self) -> &'static str { "progressive-shortlist" }
    fn search(&self, postings: &[Posting], query: &[Scalar], nprobe: usize, k: usize) -> Vec<Candidate> {
        let probes = pick_probes(postings, query, nprobe);
        let mut topk = TopK::new(k);
        // Per-list state: (pi, q_minus_c, qcn, qc, cursor).
        struct ListState {
            pi: usize,
            q_minus_c: Vec<Scalar>,
            qcn: Scalar,
            qc: Scalar,
            cursor: usize,
        }
        let mut states: Vec<ListState> = probes.iter().map(|&pi| {
            let qmc = sub(query, &postings[pi].centroid);
            let qcn = norm(&qmc);
            let qc = -(qcn * qcn);
            ListState { pi, q_minus_c: qmc, qcn, qc, cursor: 0 }
        }).collect();

        // Phase 1: scan prefix of m_init per list (or until norm-abandon).
        let m_init = self.m_init;
        for s in &mut states {
            let p = &postings[s.pi];
            let end = m_init.min(p.order.len());
            while s.cursor < end {
                let i = s.cursor;
                let rn = p.max_norm_tail[i];
                let ub = s.qc + 2.0 * s.qcn * rn - rn * rn;
                if ub < topk.threshold() { break; }
                let pos = p.order[i] as usize;
                let sc = score_from_residual(&s.q_minus_c, s.qc, &p.residuals[pos]);
                topk.push(Candidate { id: p.ids[pos], score: sc });
                s.cursor += 1;
            }
        }

        // Phase 2: revisit lists whose tail upper bound can still improve top-k.
        // Order revisits by best remaining upper bound (max first).
        loop {
            let mut best: Option<(usize, Scalar)> = None;
            for (idx, s) in states.iter().enumerate() {
                let p = &postings[s.pi];
                if s.cursor >= p.order.len() { continue; }
                let rn = p.max_norm_tail[s.cursor];
                let ub = s.qc + 2.0 * s.qcn * rn - rn * rn;
                if ub <= topk.threshold() { continue; }
                match best {
                    None => best = Some((idx, ub)),
                    Some((_, bu)) if ub > bu => best = Some((idx, ub)),
                    _ => {}
                }
            }
            let Some((idx, _)) = best else { break };
            // Scan a chunk from this list: continue until its own upper bound
            // falls below threshold OR we scan `m_init` more.
            let s = &mut states[idx];
            let p = &postings[s.pi];
            let chunk_end = (s.cursor + m_init).min(p.order.len());
            while s.cursor < chunk_end {
                let i = s.cursor;
                let rn = p.max_norm_tail[i];
                let ub = s.qc + 2.0 * s.qcn * rn - rn * rn;
                if ub < topk.threshold() { break; }
                let pos = p.order[i] as usize;
                let sc = score_from_residual(&s.q_minus_c, s.qc, &p.residuals[pos]);
                topk.push(Candidate { id: p.ids[pos], score: sc });
                s.cursor += 1;
            }
        }

        topk.into_sorted()
    }
}

#[inline]
fn sub(a: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
    let mut r = vec![0.0; a.len()];
    for i in 0..a.len() { r[i] = a[i] - b[i]; }
    r
}

/// Pick the top `nprobe` postings by score `-||q - c||^2`.
pub fn pick_probes(postings: &[Posting], query: &[Scalar], nprobe: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, Scalar)> = postings.iter().enumerate()
        .map(|(i, p)| (i, neg_l2sq(query, &p.centroid)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.iter().take(nprobe).map(|x| x.0).collect()
}

// ---------- Index build helpers ----------

/// Simple k-means (Lloyd) to produce IVF centroids. Deterministic given
/// `seed`: uses the hashed splitmix64 PRNG.
pub fn kmeans(vectors: &[Vec<Scalar>], k: usize, iters: usize, seed: u64) -> Vec<Vec<Scalar>> {
    assert!(!vectors.is_empty());
    let dim = vectors[0].len();
    let mut rng = SplitMix64 { s: seed };
    // kmeans++ init
    let mut centroids: Vec<Vec<Scalar>> = Vec::with_capacity(k);
    centroids.push(vectors[(rng.next_u64() as usize) % vectors.len()].clone());
    for _ in 1..k {
        let mut dists: Vec<Scalar> = vectors.iter().map(|v| {
            centroids.iter()
                .map(|c| { let d = neg_l2sq(v, c); -d })
                .fold(Scalar::INFINITY, Scalar::min)
        }).collect();
        let total: Scalar = dists.iter().sum();
        if total <= 0.0 {
            centroids.push(vectors[(rng.next_u64() as usize) % vectors.len()].clone());
            continue;
        }
        let r: Scalar = (rng.next_f32()) * total;
        let mut acc = 0.0;
        let mut pick = 0;
        for (i, d) in dists.iter_mut().enumerate() { acc += *d; if acc >= r { pick = i; break; } }
        centroids.push(vectors[pick].clone());
    }
    // Lloyd iterations
    for _ in 0..iters {
        let mut sums = vec![vec![0.0 as Scalar; dim]; k];
        let mut counts = vec![0u32; k];
        for v in vectors {
            let (ci, _) = assign(v, &centroids);
            for d in 0..dim { sums[ci][d] += v[d]; }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] == 0 {
                centroids[ci] = vectors[(rng.next_u64() as usize) % vectors.len()].clone();
                continue;
            }
            let inv = 1.0 / counts[ci] as Scalar;
            for d in 0..dim { centroids[ci][d] = sums[ci][d] * inv; }
        }
    }
    centroids
}

#[inline]
pub fn assign(v: &[Scalar], centroids: &[Vec<Scalar>]) -> (usize, Scalar) {
    let mut best = 0usize;
    let mut best_s = Scalar::NEG_INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let s = neg_l2sq(v, c);
        if s > best_s { best_s = s; best = i; }
    }
    (best, best_s)
}

pub fn build_postings(vectors: &[Vec<Scalar>], centroids: Vec<Vec<Scalar>>) -> Vec<Posting> {
    let k = centroids.len();
    let mut buckets: Vec<(Vec<u32>, Vec<Vec<Scalar>>)> = (0..k).map(|_| (Vec::new(), Vec::new())).collect();
    for (i, v) in vectors.iter().enumerate() {
        let (ci, _) = assign(v, &centroids);
        buckets[ci].0.push(i as u32);
        buckets[ci].1.push(v.clone());
    }
    centroids.into_iter().zip(buckets.into_iter()).map(|(c, (ids, vs))| Posting::build(c, ids, &vs)).collect()
}

// ---------- SplitMix64 PRNG (dep-free, deterministic) ----------

pub struct SplitMix64 { pub s: u64 }
impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        // uniform in [0,1)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn next_gauss(&mut self) -> f32 {
        // Box–Muller
        let u1 = (self.next_f32().max(1e-9_f32)).min(1.0 - 1e-9_f32);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Gaussian-mixture synthetic dataset: `k_clusters` Gaussian centers in `dim`
/// space, each with `per_cluster` points. Separation grows with `sep`.
pub fn gen_mixture(n: usize, dim: usize, k_clusters: usize, sep: f32, seed: u64) -> Vec<Vec<f32>> {
    let per_cluster = n / k_clusters;
    let mut rng = SplitMix64 { s: seed };
    let mut centers = Vec::with_capacity(k_clusters);
    for _ in 0..k_clusters {
        let mut c = vec![0.0; dim];
        for d in 0..dim { c[d] = rng.next_gauss() * sep; }
        centers.push(c);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % k_clusters];
        let _ = per_cluster;
        let mut v = vec![0.0; dim];
        for d in 0..dim { v[d] = c[d] + rng.next_gauss(); }
        out.push(v);
    }
    out
}

pub fn exact_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Candidate> {
    let mut v: Vec<(u32, f32)> = vectors.iter().enumerate()
        .map(|(i, x)| (i as u32, neg_l2sq(query, x)))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    v.iter().take(k).map(|(id, s)| Candidate { id: *id, score: *s }).collect()
}

pub fn recall(found: &[Candidate], truth: &[Candidate]) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<u32> = truth.iter().map(|c| c.id).collect();
    let hit = found.iter().filter(|c| t.contains(&c.id)).count();
    hit as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_equivalence() {
        // Residual form must match the direct score.
        let q = vec![0.1_f32, -0.2, 0.3, 0.5];
        let c = vec![0.0_f32, 0.0, 0.0, 0.0];
        let v = vec![0.2_f32, 0.1, 0.0, -0.1];
        let qmc = sub(&q, &c);
        let qc = -dot(&qmc, &qmc);
        let r = sub(&v, &c);
        let sr = score_from_residual(&qmc, qc, &r);
        let sd = neg_l2sq(&q, &v);
        assert!((sr - sd).abs() < 1e-5, "{} vs {}", sr, sd);
    }

    #[test]
    fn topk_sorts_desc() {
        let mut t = TopK::new(3);
        for (i, s) in [(0, 0.1), (1, -1.0), (2, 0.5), (3, 0.2)].iter() {
            t.push(Candidate { id: *i, score: *s });
        }
        let out = t.into_sorted();
        assert_eq!(out[0].id, 2);
        assert_eq!(out[1].id, 3);
        assert_eq!(out[2].id, 0);
    }

    #[test]
    fn scanners_agree_on_recall_floor() {
        // All three scanners share the same score function so they must be
        // exactly equivalent when probing all lists (nprobe = k_clusters)
        // with m_init large enough — this exercises the index plumbing.
        let vecs = gen_mixture(2000, 16, 32, 4.0, 42);
        let centroids = kmeans(&vecs, 32, 8, 7);
        let postings = build_postings(&vecs, centroids);
        let q = gen_mixture(1, 16, 1, 1.0, 11).into_iter().next().unwrap();
        let truth = exact_topk(&vecs, &q, 10);

        let e = Exhaustive.search(&postings, &q, 32, 10);
        let n = NormOrderedAbandon.search(&postings, &q, 32, 10);
        let p = Progressive { m_init: 10_000 }.search(&postings, &q, 32, 10);

        // Exhaustive nprobe = all centroids == exact top-k.
        assert_eq!(recall(&e, &truth), 1.0);
        // Other scanners are allowed to tie exhaustive on this trivial setting.
        assert!(recall(&n, &truth) >= 1.0 - 1e-6);
        assert!(recall(&p, &truth) >= 1.0 - 1e-6);
    }
}
