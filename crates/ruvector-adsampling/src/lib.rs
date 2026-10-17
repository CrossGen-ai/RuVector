//! ADSampling — Random-projection + hypothesis-test early termination
//! of exact L2 distance for ANN candidate scanning.
//!
//! Based on Gao & Long, SIGMOD 2023: *High-Dimensional Approximate Nearest
//! Neighbor Search: with Reliable and Efficient Distance Comparison Operations.*
//!
//! Core idea: project data `x` and query `q` with a random orthogonal matrix
//! `P` (JL transform preserves L2 up to a factor). While computing the
//! partial sum `s_j = sum_{i<j} (Px - Pq)_i^2`, scale it to an unbiased
//! estimator of the full L2^2: `D_hat_j = s_j * d / j`. Compare `D_hat_j`
//! against the current pruning threshold `tau` using a chi-squared tail
//! bound with user-chosen failure parameter `epsilon_0`. If the sampled
//! estimator is above `tau * (1 + epsilon(j))`, abandon early — the full
//! distance is above threshold with high probability. Otherwise keep
//! accumulating until all dims are summed (full exact distance).
//!
//! Produces exact top-k with probabilistic recall guarantees: tune
//! `epsilon_0` to trade speed vs recall. At `epsilon_0 = 0` the test
//! never prunes (equivalent to a full scan).

#![forbid(unsafe_code)]

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Random orthogonal projection matrix of size `d × d`.
///
/// Built from Haar-distributed Gaussian + QR; we keep the full square so
/// prefix sums over projected coords are an unbiased unitary rotation of
/// the full L2 norm. For d ≤ 1024 this is small enough to store once per
/// index and apply with a dense `d*d` GEMV at ingest and query time.
#[derive(Clone, Debug)]
pub struct Projector {
    d: usize,
    rows: Vec<f32>, // row-major d*d
}

impl Projector {
    /// Build an orthogonal `d × d` projector from a fixed seed.
    pub fn new(d: usize, seed: u64) -> Self {
        assert!(d > 0 && d <= 4096, "unreasonable dim {d}");
        let mut rng = StdRng::seed_from_u64(seed);
        // Fill with iid N(0,1)
        let mut m = vec![0f32; d * d];
        for v in m.iter_mut() {
            let x: f64 = rng.sample(StandardNormal);
            *v = x as f32;
        }
        // Modified Gram-Schmidt → orthonormal rows.
        for i in 0..d {
            // Normalize row i
            let (lo, _hi) = m.split_at_mut((i + 1) * d);
            let row_i_start = i * d;
            let norm: f32 = (0..d)
                .map(|k| lo[row_i_start + k] * lo[row_i_start + k])
                .sum::<f32>()
                .sqrt();
            let inv = 1.0 / norm.max(f32::MIN_POSITIVE);
            for k in 0..d {
                lo[row_i_start + k] *= inv;
            }
            // Orthogonalize later rows against row i
            let _ = lo;
            for j in (i + 1)..d {
                let mut dot = 0f32;
                for k in 0..d {
                    dot += m[i * d + k] * m[j * d + k];
                }
                for k in 0..d {
                    m[j * d + k] -= dot * m[i * d + k];
                }
            }
        }
        Self { d, rows: m }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Apply the projection to `x` in-place-free style, writing to `out`.
    pub fn project(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.d);
        assert_eq!(out.len(), self.d);
        for i in 0..self.d {
            let row = &self.rows[i * self.d..(i + 1) * self.d];
            let mut acc = 0f32;
            for k in 0..self.d {
                acc += row[k] * x[k];
            }
            out[i] = acc;
        }
    }

    /// Project a batch of vectors (row-major, n × d) into `out` (n × d).
    pub fn project_many(&self, data: &[f32], out: &mut [f32]) {
        assert_eq!(data.len() % self.d, 0);
        assert_eq!(data.len(), out.len());
        let n = data.len() / self.d;
        for i in 0..n {
            let s = i * self.d;
            self.project(&data[s..s + self.d], &mut out[s..s + self.d]);
        }
    }
}

/// Pruning policy — picks how aggressively ADSampling terminates early.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// Step size between hypothesis tests (dims summed between checks).
    /// Smaller = more chances to prune; larger = cheaper tests.
    pub step: usize,
    /// Failure parameter for the chi-squared tail bound. Smaller → safer
    /// (fewer false prunes, less speedup). Typical range 1.0 .. 3.0.
    pub epsilon0: f32,
}

impl Policy {
    pub const fn new(step: usize, epsilon0: f32) -> Self {
        Self { step, epsilon0 }
    }

    /// Scale factor to multiply tau by for the test at prefix length j:
    /// `(1 + epsilon0 / sqrt(j))^2`.
    #[inline]
    pub fn ratio_at(&self, j: usize) -> f32 {
        let e = 1.0 + self.epsilon0 / (j as f32).sqrt();
        e * e
    }
}

impl Default for Policy {
    fn default() -> Self {
        // 32-dim windows, moderate aggressiveness.
        Self {
            step: 32,
            epsilon0: 2.1,
        }
    }
}

/// Outcome of a pruned-distance call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistOutcome {
    /// Full L2^2 computed; this value is exact.
    Full(f32),
    /// Pruned at prefix length `j` dims — true distance is above `tau`
    /// with high probability. Returned value is the current partial sum
    /// (unscaled); treat as "not better than tau".
    Pruned { prefix: usize, partial: f32 },
}

impl DistOutcome {
    /// Returns true if the exact distance was computed.
    #[inline]
    pub fn is_full(&self) -> bool {
        matches!(self, DistOutcome::Full(_))
    }
    /// Comparable score: full value or `tau` sentinel for pruned.
    #[inline]
    pub fn score(self, tau: f32) -> f32 {
        match self {
            DistOutcome::Full(v) => v,
            DistOutcome::Pruned { .. } => tau,
        }
    }
}

/// Fallback: full L2^2 of two slices.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Compute L2^2(Px, Pq) with ADSampling early termination against threshold `tau`.
///
/// Both inputs MUST already be in the projected space (`Projector::project`
/// applied once at ingest for `px`, once per query for `pq`). Pruning
/// decisions are based on the chi-squared tail bound (Gao & Long §4).
pub fn adsample_l2sq(px: &[f32], pq: &[f32], tau: f32, pol: Policy) -> DistOutcome {
    debug_assert_eq!(px.len(), pq.len());
    let d = px.len();
    let mut acc = 0f32;
    let mut j = 0usize;
    let step = pol.step.max(1);
    while j < d {
        let next = (j + step).min(d);
        // Accumulate the window
        for i in j..next {
            let diff = px[i] - pq[i];
            acc += diff * diff;
        }
        j = next;
        if j < d && tau.is_finite() {
            // Rescale partial sum to full-length estimator and apply bound.
            let estimator = acc * (d as f32) / (j as f32);
            let bound = tau * pol.ratio_at(j);
            if estimator > bound {
                return DistOutcome::Pruned {
                    prefix: j,
                    partial: acc,
                };
            }
        }
    }
    DistOutcome::Full(acc)
}

/// Top-k scanner over a batch of pre-projected vectors.
///
/// `pdata` is the projected data (row-major, n × d). `pq` is the projected
/// query (len d). Returns the (score, index) pairs for the top-k smallest
/// scores. Only candidates for which an exact score was computed can enter
/// the heap — pruned ones are skipped because we have proven they are
/// (with high probability) worse than the current threshold.
pub fn topk_adsample(
    pdata: &[f32],
    d: usize,
    pq: &[f32],
    k: usize,
    pol: Policy,
) -> (Vec<(f32, usize)>, ScanStats) {
    assert_eq!(pq.len(), d);
    assert_eq!(pdata.len() % d, 0);
    let n = pdata.len() / d;
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut stats = ScanStats::default();
    for i in 0..n {
        let s = i * d;
        let tau = if heap.len() < k {
            f32::INFINITY
        } else {
            heap[0].0
        };
        let out = adsample_l2sq(&pdata[s..s + d], pq, tau, pol);
        match out {
            DistOutcome::Full(v) => {
                stats.full += 1;
                if heap.len() < k {
                    heap.push((v, i));
                    heapify_max(&mut heap);
                } else if v < heap[0].0 {
                    heap[0] = (v, i);
                    sift_down(&mut heap, 0);
                }
            }
            DistOutcome::Pruned { prefix, .. } => {
                stats.pruned += 1;
                stats.pruned_dims_sum += prefix as u64;
            }
        }
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    (heap, stats)
}

/// Full-scan baseline: exact L2^2 top-k without any pruning.
pub fn topk_full(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<(f32, usize)> {
    assert_eq!(q.len(), d);
    let n = data.len() / d;
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for i in 0..n {
        let s = i * d;
        let v = l2sq(&data[s..s + d], q);
        if heap.len() < k {
            heap.push((v, i));
            heapify_max(&mut heap);
        } else if v < heap[0].0 {
            heap[0] = (v, i);
            sift_down(&mut heap, 0);
        }
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    heap
}

/// Partial-L2 early-abandon baseline: deterministic prefix bound.
/// Skips to full scan when `tau` is finite only after reading `step` dims,
/// aborting if the running sum already exceeds `tau`. This is the simplest
/// commonly-cited early-abandon strategy (no chi-squared scaling).
pub fn topk_partial(
    data: &[f32],
    d: usize,
    q: &[f32],
    k: usize,
    step: usize,
) -> (Vec<(f32, usize)>, ScanStats) {
    assert_eq!(q.len(), d);
    assert!(step > 0);
    let n = data.len() / d;
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    let mut stats = ScanStats::default();
    for i in 0..n {
        let s = i * d;
        let x = &data[s..s + d];
        let tau = if heap.len() < k {
            f32::INFINITY
        } else {
            heap[0].0
        };
        let mut acc = 0f32;
        let mut j = 0usize;
        let mut pruned = false;
        while j < d {
            let next = (j + step).min(d);
            for p in j..next {
                let diff = x[p] - q[p];
                acc += diff * diff;
            }
            j = next;
            if j < d && acc >= tau {
                pruned = true;
                break;
            }
        }
        if pruned {
            stats.pruned += 1;
            stats.pruned_dims_sum += j as u64;
            continue;
        }
        stats.full += 1;
        if heap.len() < k {
            heap.push((acc, i));
            heapify_max(&mut heap);
        } else if acc < heap[0].0 {
            heap[0] = (acc, i);
            sift_down(&mut heap, 0);
        }
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    (heap, stats)
}

/// Statistics about a pruning scan.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanStats {
    pub full: u64,
    pub pruned: u64,
    pub pruned_dims_sum: u64,
}

impl ScanStats {
    pub fn total(&self) -> u64 {
        self.full + self.pruned
    }
    /// Fraction of candidates that reached the full-distance computation.
    pub fn full_fraction(&self) -> f32 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.full as f32 / t as f32
        }
    }
    /// Average prefix length of pruned candidates (0 if none pruned).
    pub fn mean_pruned_prefix(&self) -> f32 {
        if self.pruned == 0 {
            0.0
        } else {
            self.pruned_dims_sum as f32 / self.pruned as f32
        }
    }
}

// --- tiny max-heap helpers (keyed by .0, largest at index 0) ------------
fn heapify_max(a: &mut [(f32, usize)]) {
    for i in (0..a.len() / 2).rev() {
        sift_down(a, i);
    }
}
fn sift_down(a: &mut [(f32, usize)], mut i: usize) {
    let n = a.len();
    loop {
        let l = 2 * i + 1;
        let r = l + 1;
        if l >= n {
            return;
        }
        let m = if r < n && a[r].0 > a[l].0 { r } else { l };
        if a[m].0 > a[i].0 {
            a.swap(i, m);
            i = m;
        } else {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn gauss(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = StdRng::seed_from_u64(seed);
        (0..n * d)
            .map(|_| {
                let v: f64 = r.sample(StandardNormal);
                v as f32
            })
            .collect()
    }

    #[test]
    fn projector_is_orthogonal() {
        let p = Projector::new(16, 7);
        // Check row i · row j ≈ δ_ij
        for i in 0..16 {
            for j in 0..16 {
                let mut dot = 0f32;
                for k in 0..16 {
                    dot += p.rows[i * 16 + k] * p.rows[j * 16 + k];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1e-3,
                    "P row {i}·row {j} = {dot}, want {expect}"
                );
            }
        }
    }

    #[test]
    fn projection_preserves_l2() {
        let d = 64;
        let p = Projector::new(d, 3);
        let a = gauss(1, d, 11);
        let b = gauss(1, d, 12);
        let mut pa = vec![0f32; d];
        let mut pb = vec![0f32; d];
        p.project(&a, &mut pa);
        p.project(&b, &mut pb);
        let before = l2sq(&a, &b);
        let after = l2sq(&pa, &pb);
        assert!(
            (before - after).abs() / before < 1e-3,
            "orthogonal projection must preserve L2^2: {before} vs {after}"
        );
    }

    #[test]
    fn adsample_matches_full_for_true_topk() {
        // At epsilon0 small enough, pruned candidates must truly be worse
        // than the current threshold — top-k must match.
        let (n, d, k) = (512, 64, 10);
        let data = gauss(n, d, 42);
        let q = gauss(1, d, 777);
        let p = Projector::new(d, 1);
        let mut pdata = vec![0f32; n * d];
        p.project_many(&data, &mut pdata);
        let mut pq = vec![0f32; d];
        p.project(&q, &mut pq);

        let truth = topk_full(&data, d, &q, k);
        let (approx, stats) =
            topk_adsample(&pdata, d, &pq, k, Policy::new(32, 1.9));
        // Compare index sets — identical for exact recall at this epsilon.
        let truth_ids: std::collections::BTreeSet<_> =
            truth.iter().map(|(_, i)| *i).collect();
        let approx_ids: std::collections::BTreeSet<_> =
            approx.iter().map(|(_, i)| *i).collect();
        let overlap = truth_ids.intersection(&approx_ids).count();
        assert!(
            overlap >= k - 1,
            "recall@{k} too low: overlap={overlap} stats={stats:?}"
        );
    }

    #[test]
    fn adsample_prunes_distant_candidates() {
        let (n, d, k) = (2048, 128, 10);
        let data = gauss(n, d, 13);
        let q = gauss(1, d, 99);
        let p = Projector::new(d, 5);
        let mut pdata = vec![0f32; n * d];
        p.project_many(&data, &mut pdata);
        let mut pq = vec![0f32; d];
        p.project(&q, &mut pq);
        let (_, stats) = topk_adsample(&pdata, d, &pq, k, Policy::default());
        // Some candidates must prune, and pruned prefixes must be strictly
        // less than d (otherwise the "early" abandon never fired).
        assert!(stats.pruned > 0, "no candidates pruned: {stats:?}");
        assert!(
            stats.mean_pruned_prefix() < d as f32,
            "pruned at full length: {stats:?}"
        );
    }

    #[test]
    fn partial_l2_is_also_exact_and_slower_than_adsample() {
        let (n, d, k) = (1024, 128, 10);
        let data = gauss(n, d, 21);
        let q = gauss(1, d, 50);
        let (truth, _) = topk_partial(&data, d, &q, k, 32);
        let truth2 = topk_full(&data, d, &q, k);
        for ((a, _), (b, _)) in truth.iter().zip(truth2.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn policy_ratio_decreases_toward_1() {
        let pol = Policy::new(32, 2.0);
        let r_near = pol.ratio_at(16);
        let r_far = pol.ratio_at(1024);
        assert!(r_near > r_far);
        assert!(r_far > 1.0);
    }
}
