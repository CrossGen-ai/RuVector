//! Integration tests for `ruvector-rpq`.

use ruvector_rpq::{kmeans, sq_l2, Pq, Quantizer, Rng, Rpq2, Rpq2Scorer, Sq8};

fn make_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut v = vec![0.0f32; n * dim];
    for x in v.iter_mut() {
        *x = rng.next_gauss();
    }
    v
}

#[test]
fn kmeans_produces_k_centroids() {
    let mut rng = Rng::new(1);
    let data = make_data(400, 8, 7);
    let cent = kmeans(&data, 8, 16, 5, &mut rng);
    assert_eq!(cent.len(), 16 * 8);
}

#[test]
fn pq_encode_decode_reasonable_error() {
    let mut rng = Rng::new(2);
    let dim = 32;
    let train = make_data(2_000, dim, 3);
    let pq = Pq::train(&train, dim, 8, 8, &mut rng);
    let mut code = vec![0u8; pq.code_bytes()];
    let mut recon = vec![0.0f32; dim];
    let mut err = 0.0f64;
    let mut norm = 0.0f64;
    for i in 0..200 {
        let x = &train[i * dim..(i + 1) * dim];
        pq.encode(x, &mut code);
        pq.reconstruct(&code, &mut recon);
        err += sq_l2(x, &recon) as f64;
        norm += sq_l2(x, &vec![0.0f32; dim]) as f64;
    }
    let ratio = err / norm;
    assert!(ratio < 0.6, "PQ reconstruction ratio too high: {ratio}");
}

#[test]
fn rpq2_beats_pq_reconstruction() {
    let mut rng = Rng::new(4);
    let dim = 32;
    let train = make_data(2_000, dim, 9);
    let pq = Pq::train(&train, dim, 8, 8, &mut rng);
    let rpq = Rpq2::train(&train, dim, 8, 8, 8, &mut rng);
    assert_eq!(pq.code_bytes(), 8);
    assert_eq!(rpq.code_bytes(), 16);

    let mut c_pq = vec![0u8; pq.code_bytes()];
    let mut c_rpq = vec![0u8; rpq.code_bytes()];
    let mut r_pq = vec![0.0f32; dim];
    let mut r1 = vec![0.0f32; dim];
    let mut r2 = vec![0.0f32; dim];
    let mut recon_rpq = vec![0.0f32; dim];

    let mut err_pq = 0.0f64;
    let mut err_rpq = 0.0f64;
    for i in 0..500 {
        let x = &train[i * dim..(i + 1) * dim];
        pq.encode(x, &mut c_pq);
        pq.reconstruct(&c_pq, &mut r_pq);
        err_pq += sq_l2(x, &r_pq) as f64;

        rpq.encode(x, &mut c_rpq);
        let (c1, c2) = c_rpq.split_at(rpq.m1());
        rpq.pq1().reconstruct(c1, &mut r1);
        rpq.pq2().reconstruct(c2, &mut r2);
        for d in 0..dim {
            recon_rpq[d] = r1[d] + r2[d];
        }
        err_rpq += sq_l2(x, &recon_rpq) as f64;
    }
    // With twice the code budget, RPQ2 must beat single-level PQ on
    // reconstruction error — this is a bit-budget-fairness sanity check,
    // not a query-time recall claim (see docs/research/... for the caveat).
    assert!(err_rpq < err_pq, "RPQ2 err {err_rpq} not < PQ err {err_pq}");
}

#[test]
fn sq8_roundtrip_bounded_error() {
    let dim = 16;
    let train = make_data(400, dim, 11);
    let sq = Sq8::train(&train, dim);
    assert_eq!(sq.code_bytes(), dim);
    let mut code = vec![0u8; dim];
    for i in 0..100 {
        let x = &train[i * dim..(i + 1) * dim];
        sq.encode(x, &mut code);
        let d = sq.adc_sq_distance(x, &code);
        assert!(d < 1e-2 * dim as f32, "sq8 self-distance too large: {d}");
    }
}

#[test]
fn adc_matches_reference() {
    let mut rng = Rng::new(5);
    let dim = 32;
    let train = make_data(1_000, dim, 6);
    let pq = Pq::train(&train, dim, 8, 8, &mut rng);
    let mut code = vec![0u8; pq.code_bytes()];
    let mut recon = vec![0.0f32; dim];
    let q = make_data(1, dim, 42);
    for i in 0..50 {
        let x = &train[i * dim..(i + 1) * dim];
        pq.encode(x, &mut code);
        pq.reconstruct(&code, &mut recon);
        let ref_d = sq_l2(&q, &recon);
        let adc_d = pq.adc_sq_distance(&q, &code);
        let rel = (ref_d - adc_d).abs() / ref_d.max(1e-6);
        assert!(rel < 1e-4, "ADC mismatch ref={ref_d} adc={adc_d}");
    }
}

#[test]
fn rpq2_bucket_scorer_matches_naive() {
    let mut rng = Rng::new(6);
    let dim = 16;
    let train = make_data(400, dim, 8);
    let rpq = Rpq2::train(&train, dim, 4, 4, 6, &mut rng);
    let n = 200;
    let db = &train[..n * dim];
    let cb = rpq.code_bytes();
    let mut codes = vec![0u8; n * cb];
    for i in 0..n {
        rpq.encode(&db[i * dim..(i + 1) * dim], &mut codes[i * cb..(i + 1) * cb]);
    }
    let scorer = Rpq2Scorer::new(&rpq, &codes);
    let query = make_data(1, dim, 999);
    let mut out = vec![0.0f32; n];
    scorer.score_all(&query, &mut out);
    for i in 0..n {
        let naive = rpq.adc_sq_distance(&query, &codes[i * cb..(i + 1) * cb]);
        let rel = (out[i] - naive).abs() / naive.max(1e-6);
        assert!(rel < 1e-4, "scorer mismatch at {i}: bucket={} naive={}", out[i], naive);
    }
}
