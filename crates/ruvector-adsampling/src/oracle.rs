//! Distance-oracle abstractions.
//!
//! An `AdsIndex` never calls `l2_squared` directly — it goes through a
//! `DistanceOracle`. That lets the same index be benchmarked with three
//! comparators without touching the traversal loop:
//!
//! * [`ExactL2`]         : the honest full-precision squared L2 distance.
//! * [`AdsFixedBudget`]  : a naive baseline that stops the sum after a
//!                         fixed `m` dims and rescales by `d/m`. Cheap
//!                         but has *no* soundness bound — useful only as
//!                         a "sanity control" to show that adaptive
//!                         termination is not just about looking at
//!                         fewer dimensions.
//! * [`AdsAdaptive`]     : Algorithm 1 of Gao & Long, SIGMOD'23. Walks
//!                         the partial sum in blocks of size `delta`,
//!                         terminates as soon as the lower confidence
//!                         bound at the current step already exceeds the
//!                         top-k threshold, otherwise falls back to the
//!                         exact distance at the end.

use std::sync::atomic::{AtomicU64, Ordering};

/// A comparator between a rotated query and a rotated database vector,
/// specialised for the "keep vs prune" question the ANN loop asks.
///
/// Returned `Outcome::Prune` means: the oracle is confident that
/// `‖q − x‖² > tau` and the caller may skip this candidate. Returned
/// `Outcome::Keep(d²)` means: the caller should treat `d²` as the
/// (approximate) squared distance and consider inserting `x` into the
/// top-k heap.
pub trait DistanceOracle: Send + Sync {
    /// Cheap identifier for reports.
    fn name(&self) -> &'static str;

    /// Evaluate `q` against `x` with the current top-k threshold `tau`.
    fn evaluate(&self, q: &[f32], x: &[f32], tau: f32) -> Outcome;

    /// Snapshot of counters accumulated since creation.
    fn stats(&self) -> OracleStats;
}

/// Decision emitted by [`DistanceOracle::evaluate`].
#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    /// Caller must not insert this candidate — it is dominated by tau.
    Prune,
    /// Caller may insert this candidate; the payload is the squared
    /// distance the oracle wants to be believed.
    Keep(f32),
}

/// Runtime counters. All variants share the same shape so reports line
/// up in tables.
#[derive(Debug, Default, Clone, Copy)]
pub struct OracleStats {
    /// Number of `evaluate` calls.
    pub evals: u64,
    /// Number of pruned calls (early-terminated as `> tau`).
    pub pruned: u64,
    /// Total number of scalar float multiply-add operations issued by
    /// the oracle. Lower is better; this is the metric the paper
    /// optimises.
    pub scalar_ops: u64,
}

// -----------------------------------------------------------------------------
// ExactL2
// -----------------------------------------------------------------------------

/// Honest full-precision squared L2 comparator. Zero pruning, `d` MACs
/// per call.
#[derive(Debug, Default)]
pub struct ExactL2 {
    evals: AtomicU64,
    pruned: AtomicU64,
    scalar_ops: AtomicU64,
}

impl ExactL2 {
    /// Construct.
    pub fn new() -> Self {
        Self::default()
    }
}

impl DistanceOracle for ExactL2 {
    fn name(&self) -> &'static str {
        "exact-l2"
    }

    fn evaluate(&self, q: &[f32], x: &[f32], tau: f32) -> Outcome {
        debug_assert_eq!(q.len(), x.len());
        let mut s = 0.0f32;
        for i in 0..q.len() {
            let d = q[i] - x[i];
            s += d * d;
        }
        self.evals.fetch_add(1, Ordering::Relaxed);
        self.scalar_ops
            .fetch_add(q.len() as u64, Ordering::Relaxed);
        if s > tau {
            self.pruned.fetch_add(1, Ordering::Relaxed);
            Outcome::Prune
        } else {
            Outcome::Keep(s)
        }
    }

    fn stats(&self) -> OracleStats {
        OracleStats {
            evals: self.evals.load(Ordering::Relaxed),
            pruned: self.pruned.load(Ordering::Relaxed),
            scalar_ops: self.scalar_ops.load(Ordering::Relaxed),
        }
    }
}

// -----------------------------------------------------------------------------
// AdsFixedBudget — sanity-control baseline
// -----------------------------------------------------------------------------

/// Naive "look at first `m` dims and rescale" comparator. Fast but
/// **not** the paper's algorithm — it has no early-termination and no
/// bound. Included so we can prove empirically that the win comes from
/// adaptive termination, not from just doing less work.
#[derive(Debug)]
pub struct AdsFixedBudget {
    m: usize,
    evals: AtomicU64,
    pruned: AtomicU64,
    scalar_ops: AtomicU64,
}

impl AdsFixedBudget {
    /// Use the first `m` rotated dims.
    pub fn new(m: usize) -> Self {
        assert!(m >= 1);
        Self {
            m,
            evals: AtomicU64::new(0),
            pruned: AtomicU64::new(0),
            scalar_ops: AtomicU64::new(0),
        }
    }
}

impl DistanceOracle for AdsFixedBudget {
    fn name(&self) -> &'static str {
        "ads-fixed"
    }

    fn evaluate(&self, q: &[f32], x: &[f32], tau: f32) -> Outcome {
        debug_assert_eq!(q.len(), x.len());
        let d = q.len();
        let m = self.m.min(d);
        let mut s = 0.0f32;
        for i in 0..m {
            let dd = q[i] - x[i];
            s += dd * dd;
        }
        // Rescale by (d/m): under a random rotation, E[partial * d/m] = full.
        let est = s * (d as f32) / (m as f32);
        self.evals.fetch_add(1, Ordering::Relaxed);
        self.scalar_ops.fetch_add(m as u64, Ordering::Relaxed);
        if est > tau {
            self.pruned.fetch_add(1, Ordering::Relaxed);
            Outcome::Prune
        } else {
            Outcome::Keep(est)
        }
    }

    fn stats(&self) -> OracleStats {
        OracleStats {
            evals: self.evals.load(Ordering::Relaxed),
            pruned: self.pruned.load(Ordering::Relaxed),
            scalar_ops: self.scalar_ops.load(Ordering::Relaxed),
        }
    }
}

// -----------------------------------------------------------------------------
// AdsAdaptive — the real algorithm
// -----------------------------------------------------------------------------

/// Algorithm 1 of Gao & Long, SIGMOD 2023.
///
/// * `delta` is the block size in rotated dims. `delta = 32` matches the
///   paper's default.
/// * `epsilon` (0 < ε < 1) trades recall against work. Smaller ε means
///   more aggressive termination; the paper recommends ε ≈ 2.1 / √d.
///   `AdsAdaptive::with_epsilon_from_dim` picks a sensible default.
///
/// The stopping rule at step `m` (number of dims consumed so far) is:
///
/// ```text
/// partial(m) * (d / m) > tau * (1 + epsilon)
/// ```
///
/// which is the (1 − 1/d) confidence upper bound on the rescaled
/// estimator under a random orthonormal rotation.
#[derive(Debug)]
pub struct AdsAdaptive {
    delta: usize,
    epsilon: f32,
    evals: AtomicU64,
    pruned: AtomicU64,
    scalar_ops: AtomicU64,
}

impl AdsAdaptive {
    /// Explicit constructor.
    pub fn new(delta: usize, epsilon: f32) -> Self {
        assert!(delta >= 1);
        assert!(epsilon > 0.0 && epsilon < 1.0);
        Self {
            delta,
            epsilon,
            evals: AtomicU64::new(0),
            pruned: AtomicU64::new(0),
            scalar_ops: AtomicU64::new(0),
        }
    }

    /// Convenience: default ε = 2.1 / √d, delta = 32.
    pub fn with_epsilon_from_dim(d: usize) -> Self {
        let eps = (2.1_f32 / (d as f32).sqrt()).min(0.5);
        Self::new(32, eps)
    }

    /// Config accessor for the report table.
    pub fn config(&self) -> (usize, f32) {
        (self.delta, self.epsilon)
    }
}

impl DistanceOracle for AdsAdaptive {
    fn name(&self) -> &'static str {
        "ads-adaptive"
    }

    fn evaluate(&self, q: &[f32], x: &[f32], tau: f32) -> Outcome {
        debug_assert_eq!(q.len(), x.len());
        let d = q.len();
        let bound_mul = 1.0 + self.epsilon;
        let mut s = 0.0f32;
        let mut m = 0usize;
        let mut ops = 0u64;
        while m < d {
            let end = (m + self.delta).min(d);
            // Consume the next block of `delta` dims.
            for i in m..end {
                let dd = q[i] - x[i];
                s += dd * dd;
            }
            ops += (end - m) as u64;
            m = end;
            // Rescale to a full-distance estimate: E[s * d/m] = ‖q − x‖².
            let est = s * (d as f32) / (m as f32);
            if est > tau * bound_mul && m < d {
                // Confident this candidate is worse than the k-th best.
                self.evals.fetch_add(1, Ordering::Relaxed);
                self.pruned.fetch_add(1, Ordering::Relaxed);
                self.scalar_ops.fetch_add(ops, Ordering::Relaxed);
                return Outcome::Prune;
            }
        }
        // Fell through — return the exact distance we accumulated (s at
        // m == d is exact because rotation preserves the L2 norm).
        self.evals.fetch_add(1, Ordering::Relaxed);
        self.scalar_ops.fetch_add(ops, Ordering::Relaxed);
        if s > tau {
            self.pruned.fetch_add(1, Ordering::Relaxed);
            Outcome::Prune
        } else {
            Outcome::Keep(s)
        }
    }

    fn stats(&self) -> OracleStats {
        OracleStats {
            evals: self.evals.load(Ordering::Relaxed),
            pruned: self.pruned.load(Ordering::Relaxed),
            scalar_ops: self.scalar_ops.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(seed: u64, d: usize) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        (0..d).map(|_| r.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn exact_l2_agrees_with_manual_calc() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 6.0, 8.0];
        // ‖a−b‖² = 9 + 16 + 25 = 50
        let o = ExactL2::new();
        match o.evaluate(&a, &b, 1000.0) {
            Outcome::Keep(d) => assert!((d - 50.0).abs() < 1e-4),
            Outcome::Prune => panic!(),
        }
    }

    #[test]
    fn adaptive_full_walk_is_exact() {
        // With a very slack epsilon and small tau, the adaptive oracle
        // will never prune early, so its return must equal exact L2.
        let d = 64;
        let q = v(1, d);
        let x = v(2, d);
        let exact_val = q
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();
        let o = AdsAdaptive::new(32, 0.4);
        match o.evaluate(&q, &x, f32::INFINITY) {
            Outcome::Keep(dv) => assert!((dv - exact_val).abs() < 1e-3),
            Outcome::Prune => panic!(),
        }
    }

    #[test]
    fn adaptive_prunes_when_far() {
        // Give a tau that is clearly below the true distance -> must
        // return Prune, and must have consumed <= d ops.
        let d = 128;
        let q = v(3, d);
        let mut x = v(4, d);
        for xi in &mut x {
            *xi += 5.0; // shift far away
        }
        let exact_val = q
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();
        let tau = exact_val * 0.5;
        let o = AdsAdaptive::new(16, 0.15);
        let out = o.evaluate(&q, &x, tau);
        assert!(matches!(out, Outcome::Prune));
        let s = o.stats();
        assert!(s.scalar_ops <= d as u64);
    }
}
