//! Criterion benchmark — encoding throughput + query latency for the three
//! quantizers at PoC scale.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_anisotropic_pq::{
    distance::topk_by_dot, synthetic, Apq, OpqApq, Pq, Quantizer,
};

fn bench_query(c: &mut Criterion) {
    let dim = 128;
    let m = 16;
    let k = 256;
    let eta = 4.0;
    let max_iter = 15;
    let seed = 7;

    let ds = synthetic::make(5_000, 1, dim, 32, seed);
    let pq = Pq::train(&ds.train, m, k, max_iter, seed).unwrap();
    let apq = Apq::train(&ds.train, m, k, eta, max_iter, seed).unwrap();
    let opq = OpqApq::train(&ds.train, m, k, eta, max_iter, seed).unwrap();

    let codes_pq = pq.encode_batch(&ds.train);
    let codes_apq = apq.encode_batch(&ds.train);
    let codes_opq = opq.encode_batch(&ds.train);
    let q = &ds.queries[0];

    let mut group = c.benchmark_group("topk_by_dot");
    group.bench_function("pq", |b| {
        b.iter(|| topk_by_dot(black_box(&pq), black_box(q), black_box(&codes_pq), 10))
    });
    group.bench_function("apq", |b| {
        b.iter(|| topk_by_dot(black_box(&apq), black_box(q), black_box(&codes_apq), 10))
    });
    group.bench_function("opq", |b| {
        b.iter(|| topk_by_dot(black_box(&opq), black_box(q), black_box(&codes_opq), 10))
    });
    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
