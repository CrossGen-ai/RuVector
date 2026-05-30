use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_matryoshka::{
    l2_normalize, recall_at_k, BruteForceFull, BruteForceLow, MatryoshkaAdaptive, Retriever,
};

fn synth(n: usize, dim: usize, alpha: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigmas: Vec<f32> = (0..dim).map(|i| (1.0f32 / (1.0 + alpha * i as f32)).sqrt()).collect();
    let normals: Vec<Normal<f32>> = sigmas.iter().map(|&s| Normal::new(0.0, s).unwrap()).collect();
    let mut data = vec![0f32; n * dim];
    for row in 0..n {
        let off = row * dim;
        for j in 0..dim {
            data[off + j] = normals[j].sample(&mut rng);
        }
        l2_normalize(&mut data[off..off + dim]);
    }
    data
}

#[test]
fn full_brute_force_returns_self_first() {
    let n = 200;
    let dim = 32;
    let data = synth(n, dim, 0.05, 7);
    let f = BruteForceFull::new(data.clone(), dim).unwrap();
    let q = data[0..dim].to_vec();
    let res = f.search(&q, 5).unwrap();
    assert_eq!(res[0].0, 0, "query equals first vector, must rank it first");
    assert!((res[0].1 - 1.0).abs() < 1e-4, "score should be ~1.0, got {}", res[0].1);
}

#[test]
fn mar_recall_dominates_prefix_only() {
    let n = 4000;
    let dim = 256;
    let low = 32;
    let k = 10;
    let alpha: f32 = 0.10;
    let data = synth(n, dim, alpha, 11);
    let f = BruteForceFull::new(data.clone(), dim).unwrap();
    let lo = BruteForceLow::new(&data, dim, low).unwrap();
    let mar = MatryoshkaAdaptive::new(data.clone(), dim, low, 16).unwrap();

    let mut rng = StdRng::seed_from_u64(99);
    let sigmas: Vec<f32> = (0..dim).map(|i| (1.0f32 / (1.0 + alpha * i as f32)).sqrt()).collect();
    let normals: Vec<Normal<f32>> = sigmas.iter().map(|&s| Normal::new(0.0, s).unwrap()).collect();

    let mut sum_low = 0.0f32;
    let mut sum_mar = 0.0f32;
    let nq = 50;
    for _ in 0..nq {
        let mut q: Vec<f32> = (0..dim).map(|j| normals[j].sample(&mut rng)).collect();
        l2_normalize(&mut q);
        let mut ql: Vec<f32> = q[..low].to_vec();
        l2_normalize(&mut ql);

        let truth = f.search(&q, k).unwrap();
        let r_low = lo.search(&ql, k).unwrap();
        let r_mar = mar.search(&q, k).unwrap();
        sum_low += recall_at_k(&truth, &r_low, k);
        sum_mar += recall_at_k(&truth, &r_mar, k);
    }
    let r_low = sum_low / nq as f32;
    let r_mar = sum_mar / nq as f32;
    assert!(r_mar > r_low, "MAR recall {} should beat prefix-only {}", r_mar, r_low);
    assert!(r_mar >= 0.70, "MAR recall@{}={} below 0.70", k, r_mar);
}

#[test]
fn dim_mismatch_errors() {
    let dim = 16;
    let data = synth(10, dim, 0.05, 1);
    let f = BruteForceFull::new(data, dim).unwrap();
    let q = vec![0.0f32; dim + 1];
    assert!(f.search(&q, 3).is_err());
}

#[test]
fn bad_low_dim_errors() {
    let dim = 16;
    let data = synth(10, dim, 0.05, 1);
    assert!(MatryoshkaAdaptive::new(data.clone(), dim, 0, 4).is_err());
    assert!(MatryoshkaAdaptive::new(data, dim, dim + 1, 4).is_err());
}

#[test]
fn resident_bytes_accounting() {
    let dim = 64;
    let n = 100;
    let data = synth(n, dim, 0.05, 3);
    let f = BruteForceFull::new(data.clone(), dim).unwrap();
    assert_eq!(f.resident_bytes(), n * dim * 4);

    let mar = MatryoshkaAdaptive::new(data, dim, 8, 4).unwrap();
    assert_eq!(mar.resident_bytes(), n * dim * 4 + n * 8 * 4);
}
