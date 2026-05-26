//! Criterion benchmark: per-query search throughput for PQ vs Anisotropic PQ.
//! Real numbers — no mocks.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_anisotropic_pq::{approx_topk, ApqQuantizer, Pq, Quantizer};

fn make(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            let nrm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.iter().map(|a| a / nrm).collect()
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let n = 5_000;
    let d = 64;
    let m = 8;
    let k = 256;
    let db = make(n, d, 1);
    let q = make(1, d, 2).pop().unwrap();
    let pq = Pq::train(&db[..2_000], m, k, 6, 1).unwrap();
    let apq = ApqQuantizer::train(&db[..2_000], m, k, 3.0, 6, 1).unwrap();
    let codes_pq: Vec<Vec<u8>> = db.iter().map(|x| pq.encode(x)).collect();
    let codes_apq: Vec<Vec<u8>> = db.iter().map(|x| apq.encode(x)).collect();

    c.bench_function("pq_topk10", |b| {
        b.iter(|| approx_topk(&pq, &q, &codes_pq, 10))
    });
    c.bench_function("apq_topk10", |b| {
        b.iter(|| approx_topk(&apq, &q, &codes_apq, 10))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
