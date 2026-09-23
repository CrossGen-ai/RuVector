//! Integration tests for the three PQ backends.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_pq_fastscan_lut16::*;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

#[test]
fn scalar_pq_finds_query_itself() {
    let d = 32;
    let n = 200;
    let data = make_data(n, d, 42);
    let idx = ScalarPqAdc::build(&data, n, d, 8, 16, 12, 7).unwrap();
    let q = &data[5 * d..6 * d];
    let out = idx.search(q, 5);
    assert!(out.iter().any(|(id, _)| *id == 5), "top-5 = {out:?}");
}

#[test]
fn lut16_scalar_and_baseline_agree_at_ks16() {
    let d = 32;
    let n = 400;
    let data = make_data(n, d, 5);
    let base = ScalarPqAdc::build(&data, n, d, 8, 16, 15, 11).unwrap();
    let fs = Lut16ScalarFastscan::build(&data, n, d, 8, 15, 11).unwrap();
    let mut agree = 0.0f32;
    for qid in 0..20 {
        let q = &data[qid * d..(qid + 1) * d];
        let a: Vec<u32> = base.search(q, 10).into_iter().map(|(i, _)| i).collect();
        let b: Vec<u32> = fs.search(q, 10).into_iter().map(|(i, _)| i).collect();
        agree += recall_at_k(&a, &b);
    }
    let mean_agree = agree / 20.0;
    assert!(mean_agree >= 0.7, "u8-LUT quantization should preserve top-10; got {mean_agree}");
}

#[test]
fn lut16_blocked_matches_lut16_scalar() {
    let d = 32;
    let n = 400;
    let data = make_data(n, d, 9);
    let scalar = Lut16ScalarFastscan::build(&data, n, d, 8, 15, 13).unwrap();
    let blocked = Lut16BlockedFastscan::build(&data, n, d, 8, 15, 13).unwrap();
    for qid in 0..10 {
        let q = &data[qid * d..(qid + 1) * d];
        let a: Vec<u32> = scalar.search(q, 10).into_iter().map(|(i, _)| i).collect();
        let b: Vec<u32> = blocked.search(q, 10).into_iter().map(|(i, _)| i).collect();
        let r = recall_at_k(&a, &b);
        assert!(r >= 0.9, "scalar and blocked FastScan must agree; got {r}");
    }
}

#[test]
fn code_footprint_shrinks_as_advertised() {
    let d = 64;
    let n = 512;
    let data = make_data(n, d, 3);
    let base = ScalarPqAdc::build(&data, n, d, 16, 256, 8, 1).unwrap();
    let fs = Lut16ScalarFastscan::build(&data, n, d, 16, 8, 1).unwrap();
    assert_eq!(base.code_bytes(), n * 16);
    assert_eq!(fs.code_bytes(), n * 8);
}

#[test]
fn exact_topk_returns_self_first() {
    let d = 16;
    let n = 100;
    let data = make_data(n, d, 2);
    let q = &data[7 * d..8 * d];
    let out = exact_topk(&data, n, d, q, 3);
    assert_eq!(out[0], 7);
}
