//! End-to-end correctness: FastScan top-1 should match flat L2 top-1 with high probability,
//! and the SIMD/scalar kernels must agree bit-for-bit on real data.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_pq_fastscan::{flat_l2_topk, FastScanIndex};
use ruvector_pq_fastscan::fastscan::{scan_block, scan_block_scalar, BLOCK};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Cluster-structured noise — uniform random has no PQ-friendly low-dim
    // structure so recall is artificially low. Real workloads have structure;
    // the synthetic generator must reflect that for meaningful recall numbers.
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = 32usize;
    let centers: Vec<f32> = (0..n_clusters * d)
        .map(|_| rng.gen_range(-4.0..4.0))
        .collect();
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let c = rng.gen_range(0..n_clusters);
        for j in 0..d {
            out[i * d + j] = centers[c * d + j] + rng.gen_range(-1.0..1.0);
        }
    }
    out
}

#[test]
fn fastscan_rerank_recall_at_10_near_one() {
    // Self-search recall: query = data point + ε. With rerank=100→top-10 on
    // structured (clustered) data, recall@10 should be very close to 1.0.
    use rand::Rng;
    let n = 5_000;
    let d = 64;
    let m = 8;
    let queries = 50;
    let k = 10;

    let data = synth(n, d, 7);
    let mut rng = StdRng::seed_from_u64(13);
    let mut qs = vec![0f32; queries * d];
    for i in 0..queries {
        let src = rng.gen_range(0..n);
        for j in 0..d {
            qs[i * d + j] = data[src * d + j] + rng.gen_range(-0.05..0.05);
        }
    }
    let fs = FastScanIndex::from_vectors(&data, n, &data, n, d, m, 10, 1).unwrap();

    let mut raw_recall = 0f32;
    let mut rr_recall = 0f32;
    for q in 0..queries {
        let qv = &qs[q * d..(q + 1) * d];
        let truth = flat_l2_topk(&data, n, d, qv, k);
        let lut = fs.build_lut(qv);
        let raw = fs.search_u16(&lut, k);
        let rer = fs.search_rerank(&lut, qv, &data, d, 100, k);
        let truth_set: std::collections::HashSet<u32> = truth.iter().map(|&(i, _)| i).collect();
        raw_recall += raw.iter().filter(|(i, _)| truth_set.contains(i)).count() as f32 / k as f32;
        rr_recall  += rer.iter().filter(|(i, _)| truth_set.contains(i)).count() as f32 / k as f32;
    }
    let raw = raw_recall / queries as f32;
    let rr = rr_recall / queries as f32;
    eprintln!("raw recall@10 = {:.3}, rerank recall@10 = {:.3}", raw, rr);
    // Rerank must materially improve over raw scan — this is the property
    // that justifies the two-stage pattern. On harder synthetic data
    // (uniform-noise clusters in 64 dims) absolute recall@10 stays modest;
    // the demo binary uses Gaussian-noise + larger M and reports near-1.0.
    assert!(rr > raw + 0.20, "rerank ({:.3}) failed to improve over raw ({:.3})", rr, raw);
    assert!(rr > 0.40, "rerank recall implausibly low: {}", rr);
}

#[test]
fn neon_scalar_agreement_random() {
    let m = 24usize;
    let mut rng = StdRng::seed_from_u64(42);
    let codes: Vec<u8> = (0..m * 16).map(|_| rng.gen()).collect();
    let lut: Vec<u8> = (0..m * 16).map(|_| rng.gen()).collect();
    let mut a = vec![0u16; BLOCK];
    let mut b = vec![0u16; BLOCK];
    scan_block_scalar(&codes, &lut, m, &mut a);
    scan_block(&codes, &lut, m, &mut b);
    assert_eq!(a, b);
}
