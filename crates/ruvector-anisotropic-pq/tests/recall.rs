//! Real recall tests against brute-force MIPS ground truth.
//! No mocks: synthetic data is generated deterministically and the brute-force
//! search defines the truth.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_anisotropic_pq::{
    approx_topk, brute_force_topk, recall_at_k, ApqQuantizer, Pq, Quantizer,
};

fn synthetic(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            let nrm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.iter().map(|a| a / nrm).collect()
        })
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..n_clusters)];
            let v: Vec<f32> = c
                .iter()
                .map(|a| a + 0.15 * normal.sample(&mut rng))
                .collect();
            let nrm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.iter().map(|a| a / nrm).collect()
        })
        .collect()
}

#[test]
fn pq_recall_above_threshold() {
    let db = synthetic(4_000, 64, 32, 7);
    let queries = synthetic(40, 64, 32, 13);
    let pq = Pq::train(&db[..2_000], 16, 256, 10, 1).unwrap();
    let codes: Vec<Vec<u8>> = db.iter().map(|x| pq.encode(x)).collect();
    let mut r = 0.0_f32;
    for q in &queries {
        let truth = brute_force_topk(q, &db, 10);
        let approx = approx_topk(&pq, q, &codes, 10);
        r += recall_at_k(&approx, &truth);
    }
    r /= queries.len() as f32;
    // PQ at d=64, M=16, K=256 on clustered unit data should clear 0.5.
    assert!(r > 0.5, "PQ R@10 too low: {r}");
}

#[test]
fn apq_recall_above_threshold() {
    let db = synthetic(4_000, 64, 32, 7);
    let queries = synthetic(40, 64, 32, 13);
    let apq = ApqQuantizer::train(&db[..2_000], 16, 256, 3.0, 10, 1).unwrap();
    let codes: Vec<Vec<u8>> = db.iter().map(|x| apq.encode(x)).collect();
    let mut r = 0.0_f32;
    for q in &queries {
        let truth = brute_force_topk(q, &db, 10);
        let approx = approx_topk(&apq, q, &codes, 10);
        r += recall_at_k(&approx, &truth);
    }
    r /= queries.len() as f32;
    assert!(r > 0.5, "APQ R@10 too low: {r}");
}

#[test]
fn apq_eta_one_matches_pq_within_tolerance() {
    // η = 1 should recover PQ behavior modulo the different k-means init.
    // We just check that recall is in the same ballpark, not bit-identical.
    let db = synthetic(2_000, 32, 16, 7);
    let queries = synthetic(30, 32, 16, 13);
    let pq = Pq::train(&db[..1_000], 4, 64, 8, 1).unwrap();
    let apq1 = ApqQuantizer::train(&db[..1_000], 4, 64, 1.0, 8, 1).unwrap();
    let codes_pq: Vec<Vec<u8>> = db.iter().map(|x| pq.encode(x)).collect();
    let codes_apq: Vec<Vec<u8>> = db.iter().map(|x| apq1.encode(x)).collect();
    let mut rp = 0.0_f32;
    let mut ra = 0.0_f32;
    for q in &queries {
        let truth = brute_force_topk(q, &db, 10);
        rp += recall_at_k(&approx_topk(&pq, q, &codes_pq, 10), &truth);
        ra += recall_at_k(&approx_topk(&apq1, q, &codes_apq, 10), &truth);
    }
    rp /= queries.len() as f32;
    ra /= queries.len() as f32;
    assert!((rp - ra).abs() < 0.10, "η=1 APQ ({ra}) drifted from PQ ({rp})");
}

#[test]
fn bytes_per_vector_is_compact() {
    let db = synthetic(500, 64, 8, 1);
    let pq = Pq::train(&db, 8, 256, 4, 1).unwrap();
    assert_eq!(pq.bytes_per_vector(), 8); // M=8, K=256 → 1 byte/subspace
}

#[test]
fn dim_mismatch_errors() {
    let db = synthetic(50, 32, 4, 1);
    assert!(Pq::train(&db, 5, 16, 2, 1).is_err()); // 32 not divisible by 5
}
