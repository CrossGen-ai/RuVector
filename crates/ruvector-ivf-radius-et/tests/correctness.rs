//! Correctness tests for the LB cut.
//!
//! The soundness argument is in the ADR; these tests pin the invariants:
//!   1. At full `nprobe`, FixedNprobe and RadiusET must agree bit-for-bit.
//!   2. RadiusET must match brute-force at recall ≈ 1.0 on well-trained
//!      IVF (this is the regression test that caught the "sort by
//!      centroid distance" bug — recall dropped to 0.97).
//!   3. RadiusET must *actually prune* on typical queries.

use ruvector_ivf_radius_et::*;

fn make_data(n: usize, d: usize, seed: u64) -> FlatVectors {
    let mut rng = Rng::new(seed);
    let mut fv = FlatVectors::new(d);
    let modes = 8usize;
    let mut centers = vec![0.0_f32; modes * d];
    for i in 0..centers.len() {
        centers[i] = 5.0 * rng.next_gauss();
    }
    let mut buf = vec![0.0_f32; d];
    for i in 0..n {
        let m = i % modes;
        for j in 0..d {
            buf[j] = centers[m * d + j] + rng.next_gauss();
        }
        fv.push(&buf);
    }
    fv
}

#[test]
fn radius_et_matches_fixed_at_full_nprobe() {
    let vectors = make_data(2000, 32, 42);
    let idx = IvfIndex::train(vectors.clone(), 32, 8, 1);
    let mut rng = Rng::new(7);
    let mut q = vec![0.0_f32; 32];
    for _ in 0..32 {
        for j in 0..32 {
            q[j] = 5.0 * rng.next_gauss();
        }
        let full = FixedNprobe { nprobe: idx.nlist }.search(&idx, &q, 10);
        let et = RadiusET.search(&idx, &q, 10);
        assert_eq!(full.0.len(), et.0.len());
        for ((d1, i1), (d2, i2)) in full.0.iter().zip(et.0.iter()) {
            assert!((d1 - d2).abs() < 1e-4, "d1={d1} d2={d2}");
            assert_eq!(i1, i2);
        }
    }
}

#[test]
fn radius_et_achieves_full_recall_vs_brute_when_cells_cover_space() {
    let vectors = make_data(1500, 16, 9);
    let idx = IvfIndex::train(vectors.clone(), 24, 10, 2);
    let mut rng = Rng::new(99);
    let mut q = vec![0.0_f32; 16];
    let mut sum = 0.0_f32;
    let nq = 20;
    for _ in 0..nq {
        for j in 0..16 {
            q[j] = 5.0 * rng.next_gauss();
        }
        let gt = brute_force_topk(&vectors, &q, 10);
        let (got, _) = RadiusET.search(&idx, &q, 10);
        sum += recall_at_k(&gt, &got);
    }
    let r = sum / nq as f32;
    assert!(r > 0.999, "recall {}", r);
}

#[test]
fn radius_et_prunes_cells() {
    let vectors = make_data(3000, 24, 11);
    let idx = IvfIndex::train(vectors.clone(), 32, 10, 3);
    let mut rng = Rng::new(123);
    let mut q = vec![0.0_f32; 24];
    for j in 0..24 {
        q[j] = 5.0 * rng.next_gauss();
    }
    let (_, s_fixed) = FixedNprobe { nprobe: idx.nlist }.search(&idx, &q, 10);
    let (_, s_et) = RadiusET.search(&idx, &q, 10);
    assert!(
        s_et.cells_visited < s_fixed.cells_visited,
        "ET={} FIXED={}",
        s_et.cells_visited,
        s_fixed.cells_visited
    );
}
