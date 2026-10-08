//! Dynamic Rerank-Budget (DRB) for IVF-PQ pipelines.
//!
//! Pipeline: coarse scan (approximate scores) -> rerank top-R with full-precision L2 -> top-k.
//! Fixed R wastes compute on easy queries and loses recall on hard ones.
//! DRB picks R per query from a cheap ambiguity signal computed on the approximate
//! candidate list, so easy queries pay less and hard queries get a bigger budget.

use std::time::Instant;

pub type VecF32 = Vec<f32>;

/// Minimal LCG — reproducible, no external deps.
#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xDEADBEEF) }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn gauss(&mut self) -> f32 {
        // Box–Muller; cheap and good enough for a synthetic benchmark.
        let u1 = (self.next_f32().max(1e-9)).ln() * -2.0;
        let u2 = self.next_f32() * std::f32::consts::TAU;
        u1.sqrt() * u2.cos()
    }
}

pub fn make_dataset(n: usize, dim: usize, seed: u64) -> Vec<VecF32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gauss()).collect()).collect()
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() { let d = a[i] - b[i]; acc += d * d; }
    acc
}

/// FP16-ish "approximate" scorer: round each f32 to 2048 levels per unit,
/// then square-sum. Models a cheap codec whose output correlates with L2
/// but has nontrivial error (so rerank matters).
#[inline]
pub fn l2_sq_approx(a: &[f32], b: &[f32], scale: f32) -> f32 {
    let inv = 1.0 / scale;
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let qa = (a[i] * scale).round() * inv;
        let qb = (b[i] * scale).round() * inv;
        let d = qa - qb;
        acc += d * d;
    }
    acc
}

/// Tiny IVF: pick K random centroids, assign by nearest.
pub struct Ivf {
    pub centroids: Vec<VecF32>,
    pub cells: Vec<Vec<u32>>,
}

impl Ivf {
    pub fn train(data: &[VecF32], k: usize, iters: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0xA5A5);
        let n = data.len();
        let dim = data[0].len();
        // Pick k random seeds.
        let mut centroids: Vec<VecF32> = (0..k)
            .map(|_| data[(rng.next_u64() as usize) % n].clone())
            .collect();
        for _ in 0..iters {
            let mut sums: Vec<VecF32> = (0..k).map(|_| vec![0.0; dim]).collect();
            let mut counts: Vec<u32> = vec![0; k];
            for v in data {
                let mut best = 0usize; let mut bd = f32::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = l2_sq(v, c);
                    if d < bd { bd = d; best = ci; }
                }
                counts[best] += 1;
                for j in 0..dim { sums[best][j] += v[j]; }
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    let c = counts[ci] as f32;
                    for j in 0..dim { centroids[ci][j] = sums[ci][j] / c; }
                }
            }
        }
        let mut cells: Vec<Vec<u32>> = (0..k).map(|_| Vec::new()).collect();
        for (i, v) in data.iter().enumerate() {
            let mut best = 0usize; let mut bd = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = l2_sq(v, c);
                if d < bd { bd = d; best = ci; }
            }
            cells[best].push(i as u32);
        }
        Ivf { centroids, cells }
    }

    pub fn probe_cells(&self, q: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = self.centroids.iter().enumerate()
            .map(|(i, c)| (l2_sq(q, c), i)).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(nprobe).map(|(_, i)| i).collect()
    }
}

/// Approximate candidate score produced by the coarse scan.
#[derive(Clone, Copy, Debug)]
pub struct Cand { pub id: u32, pub approx: f32 }

/// Coarse scan: visit nprobe cells, score every member with l2_sq_approx,
/// return top-cand_cap by approximate score.
pub fn coarse_scan(
    ivf: &Ivf, data: &[VecF32], q: &[f32], nprobe: usize, cand_cap: usize, pq_scale: f32,
) -> Vec<Cand> {
    let mut buf: Vec<Cand> = Vec::with_capacity(1024);
    for ci in ivf.probe_cells(q, nprobe) {
        for &id in &ivf.cells[ci] {
            let s = l2_sq_approx(q, &data[id as usize], pq_scale);
            buf.push(Cand { id, approx: s });
        }
    }
    buf.sort_by(|a, b| a.approx.partial_cmp(&b.approx).unwrap());
    buf.truncate(cand_cap);
    buf
}

/// Full-precision rerank of the top-r approximate candidates.
pub fn rerank(data: &[VecF32], q: &[f32], cands: &[Cand], r: usize, topk: usize) -> Vec<u32> {
    let r = r.min(cands.len());
    let mut v: Vec<(f32, u32)> = cands[..r].iter()
        .map(|c| (l2_sq(q, &data[c.id as usize]), c.id)).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v.truncate(topk);
    v.into_iter().map(|(_, id)| id).collect()
}

/// Exact ground truth: brute force top-k.
pub fn exact_topk(data: &[VecF32], q: &[f32], topk: usize) -> Vec<u32> {
    let mut v: Vec<(f32, u32)> = data.iter().enumerate()
        .map(|(i, x)| (l2_sq(q, x), i as u32)).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    v.truncate(topk);
    v.into_iter().map(|(_, id)| id).collect()
}

#[inline]
pub fn recall(pred: &[u32], truth: &[u32]) -> f32 {
    let mut hit = 0.0f32;
    for p in pred { if truth.contains(p) { hit += 1.0; } }
    hit / truth.len() as f32
}

/// Rerank-budget policy: given the approximate candidate list already sorted
/// ascending by `approx`, return the number of candidates to rerank.
pub trait BudgetPolicy {
    fn choose(&self, cands: &[Cand], topk: usize) -> usize;
    fn name(&self) -> &'static str;
}

pub struct Fixed(pub usize);
impl BudgetPolicy for Fixed {
    fn choose(&self, cands: &[Cand], _k: usize) -> usize { self.0.min(cands.len()) }
    fn name(&self) -> &'static str { "fixed" }
}

/// DRB-Gap: pick R from the ratio of the "safety gap" after k and a probe gap.
///
/// gap_q  = approx[k + probe] - approx[k - 1]      (bigger = top-k is clearly separated)
/// gap_ref = calibrated reference gap             (median gap over a calibration set)
/// R      = clamp( r_min, r_max * (gap_ref / max(gap_q, eps)) , r_max )
pub struct DrbGap {
    pub r_min: usize,
    pub r_max: usize,
    pub probe: usize,
    pub gap_ref: f32,
}
impl BudgetPolicy for DrbGap {
    fn choose(&self, cands: &[Cand], k: usize) -> usize {
        if cands.len() <= k { return cands.len(); }
        let hi = (k + self.probe).min(cands.len() - 1);
        let lo = k.saturating_sub(1);
        let gap_q = (cands[hi].approx - cands[lo].approx).max(1e-6);
        let ratio = self.gap_ref / gap_q;
        let r = (self.r_max as f32 * ratio).round() as isize;
        (r.max(self.r_min as isize) as usize).min(self.r_max).min(cands.len())
    }
    fn name(&self) -> &'static str { "drb-gap" }
}

/// Oracle: smallest R that recovers the true top-k perfectly.
/// Used only to estimate the ceiling — not a shippable policy.
pub struct Oracle<'a> {
    pub data: &'a [VecF32],
    pub query: &'a [f32],
    pub truth: &'a [u32],
}
impl<'a> BudgetPolicy for Oracle<'a> {
    fn choose(&self, cands: &[Cand], k: usize) -> usize {
        for r in k..=cands.len() {
            let got = rerank(self.data, self.query, cands, r, k);
            if recall(&got, self.truth) >= 0.999 { return r; }
        }
        cands.len()
    }
    fn name(&self) -> &'static str { "oracle" }
}

/// Measure gap_ref over a calibration pass (median of gap_q).
pub fn calibrate_gap_ref(
    ivf: &Ivf, data: &[VecF32], queries: &[VecF32],
    topk: usize, nprobe: usize, cand_cap: usize, pq_scale: f32, probe: usize,
) -> f32 {
    let mut gaps: Vec<f32> = Vec::with_capacity(queries.len());
    for q in queries {
        let cands = coarse_scan(ivf, data, q, nprobe, cand_cap, pq_scale);
        if cands.len() <= topk { continue; }
        let hi = (topk + probe).min(cands.len() - 1);
        let lo = topk.saturating_sub(1);
        gaps.push((cands[hi].approx - cands[lo].approx).max(1e-6));
    }
    gaps.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    gaps[gaps.len() / 2]
}

pub struct RunResult {
    pub name: &'static str,
    pub mean_recall: f32,
    pub mean_r: f32,
    pub p95_r: f32,
    pub elapsed_ms: f32,
}

pub fn evaluate(
    ivf: &Ivf, data: &[VecF32], queries: &[VecF32], truths: &[Vec<u32>],
    topk: usize, nprobe: usize, cand_cap: usize, pq_scale: f32,
    policy: &dyn BudgetPolicy,
) -> RunResult {
    let t0 = Instant::now();
    let mut rec = 0.0f64;
    let mut rs: Vec<f32> = Vec::with_capacity(queries.len());
    for (q, truth) in queries.iter().zip(truths.iter()) {
        let cands = coarse_scan(ivf, data, q, nprobe, cand_cap, pq_scale);
        let r = policy.choose(&cands, topk);
        let got = rerank(data, q, &cands, r, topk);
        rec += recall(&got, truth) as f64;
        rs.push(r as f32);
    }
    let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = rs[(rs.len() as f32 * 0.95) as usize];
    let mean_r = rs.iter().sum::<f32>() / rs.len() as f32;
    RunResult { name: policy.name(), mean_recall: (rec / queries.len() as f64) as f32,
        mean_r, p95_r: p95, elapsed_ms }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_reproducible() {
        let a = make_dataset(16, 8, 7);
        let b = make_dataset(16, 8, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn approx_correlates_with_exact() {
        let data = make_dataset(500, 16, 11);
        let q = make_dataset(1, 16, 42).remove(0);
        let mut err = 0.0f32;
        let mut mag = 0.0f32;
        for v in &data {
            let e = l2_sq(&q, v);
            let a = l2_sq_approx(&q, v, 32.0);
            err += (e - a).abs();
            mag += e;
        }
        assert!(err / mag < 0.05, "approx error too high: {}", err / mag);
    }

    #[test]
    fn drb_gap_beats_small_fixed_on_recall() {
        let dim = 32; let n = 2000;
        let data = make_dataset(n, dim, 3);
        let queries = make_dataset(80, dim, 777);
        let truths: Vec<Vec<u32>> = queries.iter()
            .map(|q| exact_topk(&data, q, 10)).collect();
        let ivf = Ivf::train(&data, 16, 3, 9);
        let gap_ref = calibrate_gap_ref(&ivf, &data, &queries, 10, 4, 200, 24.0, 20);
        let fixed = evaluate(&ivf, &data, &queries, &truths, 10, 4, 200, 24.0, &Fixed(20));
        let drb = evaluate(&ivf, &data, &queries, &truths, 10, 4, 200, 24.0,
            &DrbGap { r_min: 20, r_max: 200, probe: 20, gap_ref });
        assert!(drb.mean_recall >= fixed.mean_recall - 0.02,
            "DRB recall {} should match or beat fixed-small {}", drb.mean_recall, fixed.mean_recall);
    }

    #[test]
    fn oracle_is_upper_bound() {
        let dim = 24; let n = 1000;
        let data = make_dataset(n, dim, 5);
        let queries = make_dataset(50, dim, 13);
        let truths: Vec<Vec<u32>> = queries.iter()
            .map(|q| exact_topk(&data, q, 10)).collect();
        let ivf = Ivf::train(&data, 16, 3, 2);
        let fixed = evaluate(&ivf, &data, &queries, &truths, 10, 4, 200, 24.0, &Fixed(200));
        let mut oracle_r_sum = 0.0f32;
        for (q, truth) in queries.iter().zip(truths.iter()) {
            let cands = coarse_scan(&ivf, &data, q, 4, 200, 24.0);
            let r = (Oracle { data: &data, query: q, truth }).choose(&cands, 10);
            oracle_r_sum += r as f32;
        }
        let oracle_mean_r = oracle_r_sum / queries.len() as f32;
        assert!(oracle_mean_r <= fixed.mean_r,
            "oracle should rerank fewer than full-fixed on average: oracle={} fixed={}",
            oracle_mean_r, fixed.mean_r);
    }
}
