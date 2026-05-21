use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_anisotropic_pq::{AnisotropicPq, Opq, Pq, Quantizer};

fn gen(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng) as f32).collect())
        .collect()
}

fn bench(c: &mut Criterion) {
    let d = 64;
    let m = 8;
    let k = 256;
    let train = gen(2_000, d, 1);
    let db = gen(4_000, d, 2);
    let q = gen(1, d, 3).pop().unwrap();

    let pq = Pq::train(&train, m, k, 20, 0).unwrap();
    let opq = Opq::train(&train, m, k, 15, 2, 0).unwrap();
    let apq = AnisotropicPq::train(&train, m, k, 4.0, 15, 2, 0).unwrap();

    let pq_codes: Vec<Vec<u8>> = db.iter().map(|v| pq.encode(v)).collect();
    let opq_codes: Vec<Vec<u8>> = db.iter().map(|v| opq.encode(v)).collect();
    let apq_codes: Vec<Vec<u8>> = db.iter().map(|v| apq.encode(v)).collect();

    c.bench_function("pq_scan_4k", |b| {
        b.iter(|| {
            let mut total = 0f32;
            for code in &pq_codes {
                total += pq.asymmetric_score(black_box(&q), code);
            }
            black_box(total)
        })
    });
    c.bench_function("opq_scan_4k", |b| {
        b.iter(|| {
            let mut total = 0f32;
            for code in &opq_codes {
                total += opq.asymmetric_score(black_box(&q), code);
            }
            black_box(total)
        })
    });
    c.bench_function("apq_scan_4k", |b| {
        b.iter(|| {
            let mut total = 0f32;
            for code in &apq_codes {
                total += apq.asymmetric_score(black_box(&q), code);
            }
            black_box(total)
        })
    });

    c.bench_function("pq_encode_1", |b| b.iter(|| pq.encode(black_box(&db[0]))));
    c.bench_function("opq_encode_1", |b| b.iter(|| opq.encode(black_box(&db[0]))));
    c.bench_function("apq_encode_1", |b| b.iter(|| apq.encode(black_box(&db[0]))));
}

criterion_group!(benches, bench);
criterion_main!(benches);
