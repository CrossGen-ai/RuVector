//! Real correctness tests (no mocks): AQ must reconstruct at least as
//! well as PQ at equal bitrate, and beam=8 must not be worse than beam=1
//! on a low-rank synthetic dataset.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_aq_beam::{
    brute_force_topk, recall_at_k, AqBeamIndex, AqParams, PqIndex, PqParams, Quantizer,
};

fn make_dataset(n: usize, d: usize, rank: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let basis: Vec<Vec<f32>> =
        (0..rank).map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect();
    (0..n)
        .map(|_| {
            let coeffs: Vec<f32> = (0..rank).map(|_| normal.sample(&mut rng)).collect();
            let mut v = vec![0.0f32; d];
            for (r, c) in coeffs.iter().enumerate() {
                for j in 0..d { v[j] += c * basis[r][j]; }
            }
            v
        })
        .collect()
}

fn mse<Q: Quantizer + ?Sized>(q: &Q, xs: &[Vec<f32>], codes: &[Vec<u8>]) -> f64 {
    let mut acc = 0.0f64;
    for (i, code) in codes.iter().enumerate() {
        let r = q.reconstruct(code);
        for j in 0..xs[i].len() {
            let dd = (xs[i][j] - r[j]) as f64;
            acc += dd * dd;
        }
    }
    acc / (codes.len() as f64 * xs[0].len() as f64)
}

#[test]
fn aq_reconstruction_beats_pq_on_correlated_data() {
    let data = make_dataset(2_000, 64, 8, 7);
    let pq = PqIndex::train(&data, PqParams { m: 4, k: 256, kmeans_iters: 10, seed: 1 })
        .expect("pq");
    let aq = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 8, kmeans_iters: 10, seed: 1 },
    )
    .expect("aq");
    let pq_codes = pq.encode_dataset(&data);
    let aq_codes = aq.encode_dataset(&data);
    let pq_mse = mse(&pq, &data, &pq_codes);
    let aq_mse = mse(&aq, &data, &aq_codes);
    println!("pq_mse={pq_mse} aq_mse={aq_mse}");
    assert!(
        aq_mse < pq_mse,
        "expected AQ to reconstruct better on rank-8 data: pq={pq_mse} aq={aq_mse}"
    );
}

#[test]
fn beam_search_helps_over_greedy() {
    let data = make_dataset(1_000, 32, 6, 11);
    let aq1 = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 1, kmeans_iters: 8, seed: 2 },
    )
    .expect("aq1");
    let aq8 = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 8, kmeans_iters: 8, seed: 2 },
    )
    .expect("aq8");
    let c1 = aq1.encode_dataset(&data);
    let c8 = aq8.encode_dataset(&data);
    let m1 = mse(&aq1, &data, &c1);
    let m8 = mse(&aq8, &data, &c8);
    println!("beam1_mse={m1} beam8_mse={m8}");
    // Beam=8 uses the same codebook init as beam=1 (same seed), so it can
    // only match-or-improve the greedy encoding.
    assert!(m8 <= m1 * 1.001, "beam=8 must not be worse than beam=1");
}

#[test]
fn recall_at_10_is_reasonable() {
    let data = make_dataset(1_500, 32, 6, 13);
    let queries = make_dataset(50, 32, 6, 17);
    let truth: Vec<_> = queries.iter().map(|q| brute_force_topk(q, &data, 10)).collect();
    let aq = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 8, kmeans_iters: 10, seed: 3 },
    )
    .expect("aq");
    let codes = aq.encode_dataset(&data);
    let mut r = 0.0f32;
    for (qi, qv) in queries.iter().enumerate() {
        let got = aq.search(qv, &codes, 10);
        r += recall_at_k(&got, &truth[qi]);
    }
    let recall = r / queries.len() as f32;
    println!("recall@10={recall}");
    // A 32-bit code (M=4, K=256) on rank-6 data should keep recall well
    // above chance and give a meaningful signal.
    assert!(recall > 0.4, "recall@10 too low: {recall}");
}
