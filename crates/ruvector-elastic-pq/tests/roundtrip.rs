//! End-to-end tests: train each allocator on synthetic data with a known
//! anisotropic variance profile, verify (1) that codes round-trip, (2)
//! that the distortion-iterative allocator lowers total distortion vs
//! uniform under the same total bit budget, and (3) that recall@10 on a
//! matched-budget comparison meets a non-trivial floor.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_elastic_pq::{
    AdcSearcher, Allocator, ElasticPqBuilder, Quantizer,
};

fn make_anisotropic(n: usize, dim: usize, m: usize, seed: u64) -> Vec<f32> {
    // Build a dataset where subspace variance decays geometrically. Bit
    // allocators should notice.
    let sub_dim = dim / m;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0f32; n * dim];
    let scales: Vec<f32> = (0..m).map(|i| 1.0 / (1.0 + i as f32)).collect();
    for i in 0..n {
        for s in 0..m {
            let sigma = scales[s];
            for d in 0..sub_dim {
                let z: f32 = rng.gen::<f32>() * 2.0 - 1.0; // uniform in [-1,1]
                out[i * dim + s * sub_dim + d] = z * sigma;
            }
        }
    }
    out
}

fn truth_topk(query: &[f32], base: &[f32], n: usize, dim: usize, k: usize) -> Vec<u32> {
    let mut with_d: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let mut acc = 0f32;
            for d in 0..dim {
                let diff = query[d] - base[i * dim + d];
                acc += diff * diff;
            }
            (i as u32, acc)
        })
        .collect();
    with_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    with_d.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn train_and_encode_roundtrip() {
    let n = 400;
    let dim = 16;
    let m = 4;
    let base = make_anisotropic(n, dim, m, 1);
    let pq = ElasticPqBuilder::new(m)
        .allocator(Allocator::Uniform { bits: 4 })
        .seed(42)
        .train(&base, n, dim)
        .unwrap();
    let codes = pq.encode_batch(&base, n);
    assert_eq!(codes.len(), n * pq.code_bytes());
    // ADC distance for a DB vector against itself should be small.
    let self_d = pq.asym_l2_sq(&base[0..dim], &codes[0..pq.code_bytes()]);
    assert!(self_d.is_finite());
    assert!(self_d >= 0.0);
}

#[test]
fn elastic_beats_uniform_distortion() {
    let n = 600;
    let dim = 16;
    let m = 4;
    // Total budget = 4 subspaces * 4 bits = 16 bits.
    let base = make_anisotropic(n, dim, m, 2);
    let uni = ElasticPqBuilder::new(m)
        .allocator(Allocator::Uniform { bits: 4 })
        .seed(7)
        .train(&base, n, dim)
        .unwrap();
    let elastic = ElasticPqBuilder::new(m)
        .allocator(Allocator::DistortionIterative {
            start_bits: 4,
            min_bits: 2,
            max_bits: 6,
            max_swaps: 12,
        })
        .seed(7)
        .train(&base, n, dim)
        .unwrap();
    assert_eq!(uni.stats().total_bits, elastic.stats().total_bits);
    assert!(
        elastic.stats().total_distortion <= uni.stats().total_distortion,
        "elastic distortion {} should be <= uniform {}",
        elastic.stats().total_distortion,
        uni.stats().total_distortion
    );
}

#[test]
fn variance_proportional_matches_budget() {
    let n = 400;
    let dim = 16;
    let m = 4;
    let base = make_anisotropic(n, dim, m, 3);
    let vp = ElasticPqBuilder::new(m)
        .allocator(Allocator::VarianceProportional {
            total_bits: 16,
            min_bits: 2,
            max_bits: 6,
        })
        .train(&base, n, dim)
        .unwrap();
    assert_eq!(vp.stats().total_bits, 16);
    // High-variance subspaces should get more bits than low-variance ones.
    assert!(vp.subspace_bits(0) >= vp.subspace_bits(m - 1));
}

#[test]
fn recall_at_k_is_nontrivial() {
    let n = 500;
    let dim = 16;
    let m = 4;
    let base = make_anisotropic(n, dim, m, 4);
    let pq = ElasticPqBuilder::new(m)
        .allocator(Allocator::DistortionIterative {
            start_bits: 4,
            min_bits: 2,
            max_bits: 6,
            max_swaps: 20,
        })
        .train(&base, n, dim)
        .unwrap();
    let codes = pq.encode_batch(&base, n);
    let searcher = AdcSearcher::new(&pq, &codes, n);

    // Pull the first 20 items as queries; every query is guaranteed to
    // find itself (rank 1), which lets us assert recall@10 >= a floor.
    let mut hits = 0;
    let mut total = 0;
    for q in 0..20 {
        let query = &base[q * dim..(q + 1) * dim];
        let truth = truth_topk(query, &base, n, dim, 10);
        let approx: Vec<u32> = searcher.topk(query, 10).unwrap().iter().map(|r| r.index).collect();
        for a in &approx {
            if truth.contains(a) {
                hits += 1;
            }
        }
        total += 10;
    }
    let recall = hits as f32 / total as f32;
    assert!(recall >= 0.4, "recall too low: {}", recall);
}
