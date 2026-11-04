//! Binary Centroid Probe Shortlist (BCPS).
//!
//! IVF probe selection usually spends non-trivial time computing full L2
//! distances from the query to every coarse centroid. For large `nlist`
//! (65k+) this becomes a meaningful fraction of query latency.
//!
//! BCPS replaces the first pass with a cheap SIMD-popcount Hamming scan
//! over 1-bit signatures of the centroids, shortlists `M` candidates,
//! then refines them with exact L2. Signatures are built on a
//! Walsh-Hadamard rotation of the centroids so that sign bits decorrelate
//! roughly isotropic bulk mass — the same trick RaBitQ uses on data
//! vectors, applied to the centroid table.

pub mod exact;
pub mod binary;
pub mod hadamard;

use std::time::Instant;

pub trait ProbeSelector {
    /// Return the top-`n` probe ids (centroid indices) for `query`,
    /// ordered by ascending L2 distance.
    fn select(&self, query: &[f32], n: usize) -> Vec<(u32, f32)>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    pub elapsed_ns: u128,
    pub bytes_per_centroid: usize,
}

pub fn bench_once<S: ProbeSelector>(sel: &S, queries: &[Vec<f32>], n: usize) -> Metrics {
    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for q in queries {
        let out = sel.select(q, n);
        for (i, d) in out {
            sink = sink.wrapping_add(i as u64).wrapping_add(d.to_bits() as u64);
        }
    }
    let ns = t0.elapsed().as_nanos();
    // keep the compiler honest
    std::hint::black_box(sink);
    Metrics { elapsed_ns: ns / queries.len() as u128, bytes_per_centroid: 0 }
}

/// Recall@N: fraction of queries where the returned top-N set exactly
/// contains the ground-truth top-N set (order ignored).
pub fn recall_at_n<S: ProbeSelector, G: ProbeSelector>(
    sel: &S,
    truth: &G,
    queries: &[Vec<f32>],
    n: usize,
) -> f32 {
    let mut hit: usize = 0;
    let mut total: usize = 0;
    for q in queries {
        let a: Vec<u32> = sel.select(q, n).into_iter().map(|(i, _)| i).collect();
        let b: std::collections::HashSet<u32> =
            truth.select(q, n).into_iter().map(|(i, _)| i).collect();
        for id in &a {
            if b.contains(id) { hit += 1; }
        }
        total += n;
    }
    hit as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Realistic IVF-centroid distribution: unit-normalized Gaussian
    // directions (matches most production embedding sets, which are
    // L2-normalized). On the unit sphere, L2 and angular distance are
    // monotonically related, so this exercises the selector the way
    // it is used in practice.
    fn gauss_unit(nlist: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..nlist)
            .map(|_| {
                let mut v: Vec<f32> = (0..d)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect();
                let n: f32 = v.iter().map(|x: &f32| x * x).sum::<f32>().sqrt();
                if n > 0.0 { for x in &mut v { *x /= n; } }
                let _ = rng.gen::<f32>(); // keep PRNG state advancing
                v
            })
            .collect()
    }

    fn synthetic_centroids(nlist: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        gauss_unit(nlist, d, seed)
    }

    fn synthetic_queries(nq: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        gauss_unit(nq, d, seed)
    }

    #[test]
    fn exact_matches_itself() {
        let cs = synthetic_centroids(1024, 64, 1);
        let qs = synthetic_queries(16, 64, 2);
        let s = exact::ExactL2Selector::new(cs);
        let r = recall_at_n(&s, &s, &qs, 8);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn binary_shortlist_hits_most_of_truth() {
        let cs = synthetic_centroids(4096, 64, 1);
        let qs = synthetic_queries(64, 64, 2);
        let exact_ = exact::ExactL2Selector::new(cs.clone());
        // 4 stacked SRHT rotations → 256 signature bits for d=64.
        // Shortlist 256/4096 = 6%, pick top-16. This is a hard
        // worst-case: isotropic Gaussian directions on the sphere.
        let bin = binary::BinarySignatureSelector::with_rotations(cs, 256, 42, 4);
        let r = recall_at_n(&bin, &exact_, &qs, 16);
        assert!(r > 0.85, "binary selector recall too low: {}", r);
    }

    #[test]
    fn binary_selector_returns_right_count() {
        let cs = synthetic_centroids(256, 32, 1);
        let bin = binary::BinarySignatureSelector::new(cs, 64, 7);
        let q = synthetic_queries(1, 32, 9).remove(0);
        let out = bin.select(&q, 10);
        assert_eq!(out.len(), 10);
        // ensure sorted ascending
        for w in out.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
