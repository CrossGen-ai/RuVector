use criterion::{criterion_group, criterion_main, Criterion};
use ruvector_early_term::data::synthesize;
use ruvector_early_term::{FixedEf, Hnsw, HnswParams, SlopePolicy};

fn bench_search(c: &mut Criterion) {
    let n = 5_000;
    let dim = 64;
    let k = 10;
    let ef = 64;
    let ds = synthesize(n, 50, dim, 32, 0.18, k, 7);
    let mut hnsw = Hnsw::new(dim, HnswParams::default());
    hnsw.build(ds.vectors);

    c.bench_function("fixed_ef_q50", |b| {
        b.iter(|| {
            let mut p = FixedEf;
            for q in &ds.queries {
                let _ = hnsw.search(q, k, ef, &mut p);
            }
        });
    });

    c.bench_function("slope_q50", |b| {
        b.iter(|| {
            let mut p = SlopePolicy::new(8, 2e-4);
            for q in &ds.queries {
                let _ = hnsw.search(q, k, ef, &mut p);
            }
        });
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
