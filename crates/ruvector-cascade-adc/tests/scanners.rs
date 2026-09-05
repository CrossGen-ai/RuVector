//! Integration tests: build a small deterministic PQ index and verify that
//! (a) all scanners return exactly `k` results,
//! (b) cascade recall ≥ full-4bit recall (progressive precision helps),
//! (c) full-8bit is at least as good as cascade (upper bound).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_cascade_adc::{
    CascadeScanner, FullEightBitScanner, FullFourBitScanner, PqIndex, PqParams, ScanResult,
    Scanner, TrainingConfig,
};

fn gen(n: usize, d: usize, centers: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = vec![0.0f32; centers * d];
    for v in c.iter_mut() { *v = (rng.gen::<f32>() - 0.5) * 4.0; }
    let mut data = vec![0.0f32; n * d];
    for r in 0..n {
        let cc = rng.gen_range(0..centers);
        for t in 0..d {
            data[r * d + t] = c[cc * d + t] + (rng.gen::<f32>() - 0.5) * 0.4;
        }
    }
    data
}

fn brute(data: &[f32], n: usize, d: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut s: Vec<(u32, f32)> = (0..n as u32).map(|i| {
        let o = i as usize * d;
        let mut acc = 0.0f32;
        for t in 0..d { let x = data[o+t] - q[t]; acc += x*x; }
        (i, acc)
    }).collect();
    s.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(k).map(|(i,_)| i).collect()
}

fn recall(gt: &[u32], got: &[ScanResult]) -> f32 {
    let hit = got.iter().filter(|r| gt.contains(&r.id)).count();
    hit as f32 / gt.len() as f32
}

fn build_small() -> (PqIndex, Vec<f32>, Vec<f32>, usize, usize, usize) {
    let n = 4_000usize;
    let n_train = 1_000usize;
    let nq = 30usize;
    let d = 32usize;
    let m = 8usize;
    let data = gen(n, d, 32, 11);
    let train = gen(n_train, d, 32, 12);
    let queries = gen(nq, d, 32, 13);
    let idx = PqIndex::build(
        &train, n_train, &data, n,
        PqParams { d, m },
        &TrainingConfig::default(),
    );
    (idx, data, queries, nq, d, m)
}

#[test]
fn scanners_return_topk() {
    let (idx, _data, queries, _nq, d, _m) = build_small();
    let mut out = Vec::new();
    let scanners: Vec<Box<dyn Scanner>> = vec![
        Box::new(FullEightBitScanner),
        Box::new(FullFourBitScanner),
        Box::new(CascadeScanner::with_floor(0.1, 50)),
    ];
    for s in &scanners {
        s.search(&idx, &queries[0..d], 10, &mut out);
        assert_eq!(out.len(), 10, "scanner {} returned {}", s.name(), out.len());
        // distances non-decreasing
        for w in out.windows(2) {
            assert!(w[0].dist <= w[1].dist + 1e-6);
        }
    }
}

#[test]
fn cascade_recall_beats_or_matches_4bit() {
    let (idx, data, queries, nq, d, _m) = build_small();
    let k = 10;
    let full4 = FullFourBitScanner;
    let full8 = FullEightBitScanner;
    let cascade = CascadeScanner::with_floor(0.10, 80);

    let mut r4 = 0.0f32;
    let mut rc = 0.0f32;
    let mut r8 = 0.0f32;
    let mut out = Vec::new();
    for q in 0..nq {
        let gt = brute(&data, 4_000, d, &queries[q*d..(q+1)*d], k);
        full4.search(&idx, &queries[q*d..(q+1)*d], k, &mut out); r4 += recall(&gt, &out);
        cascade.search(&idx, &queries[q*d..(q+1)*d], k, &mut out); rc += recall(&gt, &out);
        full8.search(&idx, &queries[q*d..(q+1)*d], k, &mut out); r8 += recall(&gt, &out);
    }
    r4 /= nq as f32;
    rc /= nq as f32;
    r8 /= nq as f32;
    println!("recall: full4={:.4} cascade={:.4} full8={:.4}", r4, rc, r8);
    // Cascade must be within a small tolerance of 8-bit (it's an approximation),
    // and must clearly beat pure 4-bit.
    assert!(rc >= r4 - 1e-6, "cascade {} < full4 {}", rc, r4);
    assert!(r8 >= rc - 0.05, "8-bit {} should upper-bound cascade {} (within 5pp)", r8, rc);
}
