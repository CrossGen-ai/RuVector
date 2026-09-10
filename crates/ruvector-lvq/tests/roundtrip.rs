//! Cross-quantizer integration tests.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_lvq::{l2_sq_f32, Lvq4, Lvq4x8, Lvq8, Quantizer};

fn make_vec(seed: u64, d: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn residual_recovers_more_than_primary() {
    // Across many random vectors, mean reconstruction error for LVQ4x8
    // should be substantially lower than LVQ4 alone.
    let d = 128;
    let q4 = Lvq4::new(d);
    let q48 = Lvq4x8::new(d);
    let n = 128;

    let mut err4 = 0.0f32;
    let mut err48 = 0.0f32;

    for s in 0..n {
        let v = make_vec(s as u64, d);
        let c4 = q4.encode(&v);
        let c48 = q48.encode(&v);
        err4 += l2_sq_f32(&v, &q4.decode(&c4));
        err48 += l2_sq_f32(&v, &q48.decode(&c48));
    }
    err4 /= n as f32;
    err48 /= n as f32;

    // Expect the residual layer to at least halve the mean squared error.
    assert!(
        err48 < err4 * 0.5,
        "residual didn't improve enough: err4={err4:.4} err4x8={err48:.4}"
    );
}

#[test]
fn bytes_per_code_ordering_holds() {
    // For d=128: LVQ4 (72 B) < LVQ8 (136 B) < LVQ4x8 (208 B) < fp32 (512 B).
    let d = 128;
    let q4 = Lvq4::new(d);
    let q8 = Lvq8::new(d);
    let q48 = Lvq4x8::new(d);
    let fp32 = d * 4;
    assert!(q4.bytes_per_code() < q8.bytes_per_code());
    assert!(q8.bytes_per_code() < q48.bytes_per_code());
    assert!(q48.bytes_per_code() < fp32);
}

#[test]
fn recall_at_10_is_reasonable_at_small_scale() {
    // Sanity: with 300 random vectors and 10 queries, LVQ8 recall@10 should
    // be > 0.90 for isotropic random data. This is not a scientific claim
    // — it just guards against a broken encoder.
    let d = 128;
    let n = 300;
    let mut rng = StdRng::seed_from_u64(2027);
    let corpus: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let q = Lvq8::new(d);
    let codes: Vec<_> = corpus.iter().map(|v| q.encode(v)).collect();

    let recall = ruvector_lvq::recall_at_k(
        queries.len(),
        corpus.len(),
        &|qi, bi| l2_sq_f32(&queries[qi], &corpus[bi]),
        |qi, bi| q.asymmetric_l2_sq(&queries[qi], &codes[bi]),
        10,
    );
    assert!(recall >= 0.90, "LVQ8 recall@10 too low: {recall}");
}
