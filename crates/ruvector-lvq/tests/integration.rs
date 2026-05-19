use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_lvq::{
    distance::{l2_sq, Metric},
    quantizer::{Encoded, Quantizer},
    recall::{recall_at, topk_exact, topk_quantized},
    LvqOne, LvqTwo, Sq8,
};

fn make_db(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| r.sample::<f32, _>(StandardNormal)).collect()).collect()
}

#[test]
fn lvq1_roundtrip_error_decreases_with_more_bits() {
    let d = 64;
    let db = make_db(500, d, 1);

    let mut q4 = LvqOne::new(d, 4).unwrap();
    q4.fit(&db).unwrap();
    let mut q8 = LvqOne::new(d, 8).unwrap();
    q8.fit(&db).unwrap();

    let mut buf = vec![0f32; d];
    let mut err4 = 0f32;
    let mut err8 = 0f32;
    for v in &db {
        let e = q4.encode(v).unwrap();
        q4.decode(&e, &mut buf);
        err4 += l2_sq(v, &buf);
        let e = q8.encode(v).unwrap();
        q8.decode(&e, &mut buf);
        err8 += l2_sq(v, &buf);
    }
    // 8-bit must reconstruct strictly better than 4-bit.
    assert!(err8 < err4, "err8={} should be < err4={}", err8, err4);
    // And recover the input to within a tight relative tolerance.
    let total_norm: f32 = db.iter().flat_map(|v| v.iter().map(|x| x * x)).sum();
    assert!(err8 / total_norm < 1e-3, "LVQ1-8 relative error {} too high", err8 / total_norm);
}

#[test]
fn lvq2_residual_strictly_better_than_primary_alone() {
    let d = 64;
    let db = make_db(400, d, 2);

    let mut q1 = LvqOne::new(d, 8).unwrap();
    q1.fit(&db).unwrap();
    let mut q2 = LvqTwo::new(d, 8, 4).unwrap();
    q2.fit(&db).unwrap();

    let mut buf = vec![0f32; d];
    let mut err1 = 0f32;
    let mut err2 = 0f32;
    for v in &db {
        let e = q1.encode(v).unwrap();
        q1.decode(&e, &mut buf);
        err1 += l2_sq(v, &buf);
        let e = q2.encode(v).unwrap();
        q2.decode(&e, &mut buf);
        err2 += l2_sq(v, &buf);
    }
    assert!(err2 < err1, "LVQ2 err {} should be < LVQ1-8 err {}", err2, err1);
}

#[test]
fn lvq1_8_recall_high() {
    let d = 64;
    let db = make_db(2_000, d, 3);
    let queries = make_db(50, d, 4);
    let k = 10;
    let gt: Vec<Vec<usize>> = queries.iter().map(|q| topk_exact(q, &db, k, Metric::L2)).collect();

    let mut q = LvqOne::new(d, 8).unwrap();
    q.fit(&db).unwrap();
    let encoded: Vec<Encoded> = db.iter().map(|v| q.encode(v).unwrap()).collect();

    let mut acc = 0f32;
    for (i, qv) in queries.iter().enumerate() {
        let approx = topk_quantized(&q, qv, &encoded, k, Metric::L2);
        acc += recall_at(&gt[i], &approx, k);
    }
    let recall = acc / queries.len() as f32;
    assert!(recall >= 0.95, "LVQ1-8 recall@10 = {} should be >= 0.95", recall);
}

#[test]
fn sq8_baseline_works() {
    let d = 32;
    let db = make_db(300, d, 5);
    let mut q = Sq8::new(d);
    q.fit(&db).unwrap();
    let e = q.encode(&db[0]).unwrap();
    let mut buf = vec![0f32; d];
    q.decode(&e, &mut buf);
    let err = l2_sq(&db[0], &buf);
    let norm: f32 = db[0].iter().map(|x| x * x).sum();
    assert!(err / norm < 0.05, "SQ8 relative error {} too high", err / norm);
}
