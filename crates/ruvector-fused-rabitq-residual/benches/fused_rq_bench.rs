//! Criterion micro-benchmarks for the three quantizers.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use ruvector_fused_rabitq_residual::{
    FusedRQR, QuantizedIndex, Quantizer, RabitQuant, Sq4Quant,
};

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
                .collect()
        })
        .collect()
}

fn bench_all(c: &mut Criterion) {
    let d = 128;
    let n = 4000;
    let vecs = gaussian(n, d, 11);
    let query = gaussian(1, d, 22).pop().unwrap();

    let rabit = QuantizedIndex::build(RabitQuant::new(d, 7), &vecs);
    let sq4 = QuantizedIndex::build(Sq4Quant::new(d, 7), &vecs);
    let fused = QuantizedIndex::build(FusedRQR::new(d, 7), &vecs);

    c.bench_function("rabitq_topk_10", |b| {
        b.iter(|| black_box(rabit.topk(black_box(&query), 10)))
    });
    c.bench_function("sq4_topk_10", |b| {
        b.iter(|| black_box(sq4.topk(black_box(&query), 10)))
    });
    c.bench_function("fused_topk_10", |b| {
        b.iter(|| black_box(fused.topk(black_box(&query), 10)))
    });

    c.bench_function("rabitq_encode_one", |b| {
        let q = RabitQuant::new(d, 7);
        b.iter(|| black_box(q.encode(black_box(&query))))
    });
    c.bench_function("fused_encode_one", |b| {
        let q = FusedRQR::new(d, 7);
        b.iter(|| black_box(q.encode(black_box(&query))))
    });
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
