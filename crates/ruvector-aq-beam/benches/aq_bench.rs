//! Criterion bench: encode + query timings for PQ vs AQ (beam 1/8/32).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_aq_beam::{AqBeamIndex, AqParams, PqIndex, PqParams, Quantizer};

fn dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

fn bench_encode(c: &mut Criterion) {
    let d = 64usize;
    let n = 500usize;
    let data = dataset(n, d, 101);
    let sample: Vec<Vec<f32>> = data.iter().take(64).cloned().collect();

    let pq = PqIndex::train(
        &data,
        PqParams { m: 4, k: 256, kmeans_iters: 6, seed: 1 },
    )
    .unwrap();
    let aq1 = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 1, kmeans_iters: 6, seed: 1 },
    )
    .unwrap();
    let aq8 = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 8, kmeans_iters: 6, seed: 1 },
    )
    .unwrap();

    c.bench_function("pq_encode_64", |b| {
        b.iter(|| black_box(pq.encode_dataset(black_box(&sample))))
    });
    c.bench_function("aq_beam1_encode_64", |b| {
        b.iter(|| black_box(aq1.encode_dataset(black_box(&sample))))
    });
    c.bench_function("aq_beam8_encode_64", |b| {
        b.iter(|| black_box(aq8.encode_dataset(black_box(&sample))))
    });
}

fn bench_search(c: &mut Criterion) {
    let d = 64usize;
    let n = 4_000usize;
    let data = dataset(n, d, 202);
    let queries = dataset(16, d, 203);

    let pq = PqIndex::train(
        &data,
        PqParams { m: 4, k: 256, kmeans_iters: 8, seed: 5 },
    )
    .unwrap();
    let aq = AqBeamIndex::train(
        &data,
        AqParams { m: 4, k: 256, beam: 8, kmeans_iters: 8, seed: 5 },
    )
    .unwrap();
    let pq_codes = pq.encode_dataset(&data);
    let aq_codes = aq.encode_dataset(&data);

    c.bench_function("pq_search", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(pq.search(black_box(q), &pq_codes, 10));
            }
        })
    });
    c.bench_function("aq_search", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(aq.search(black_box(q), &aq_codes, 10));
            }
        })
    });
}

criterion_group!(benches, bench_encode, bench_search);
criterion_main!(benches);
