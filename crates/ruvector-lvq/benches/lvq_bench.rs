use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_lvq::{
    distance::Metric,
    quantizer::{Encoded, Quantizer},
    LvqOne, LvqTwo, Sq8,
};

const N: usize = 5_000;
const D: usize = 128;

fn gen() -> Vec<Vec<f32>> {
    let mut r = StdRng::seed_from_u64(7);
    (0..N).map(|_| (0..D).map(|_| r.sample::<f32, _>(StandardNormal)).collect()).collect()
}

fn make_query() -> Vec<f32> {
    let mut r = StdRng::seed_from_u64(8);
    (0..D).map(|_| r.sample::<f32, _>(StandardNormal)).collect()
}

fn scan<Q: Quantizer>(c: &mut Criterion, name: &str, mut q: Q, db: &[Vec<f32>], query: &[f32]) {
    q.fit(db).unwrap();
    let encoded: Vec<Encoded> = db.iter().map(|v| q.encode(v).unwrap()).collect();
    let qn: f32 = query.iter().map(|x| x * x).sum();
    c.bench_function(name, |b| {
        b.iter(|| {
            let mut sum = 0f32;
            for e in &encoded {
                sum += q.distance(black_box(query), qn, e, Metric::L2);
            }
            black_box(sum)
        })
    });
}

fn bench_scans(c: &mut Criterion) {
    let db = gen();
    let query = make_query();
    scan(c, "scan/SQ8",      Sq8::new(D),                          &db, &query);
    scan(c, "scan/LVQ1-8",   LvqOne::new(D, 8).unwrap(),           &db, &query);
    scan(c, "scan/LVQ1-4",   LvqOne::new(D, 4).unwrap(),           &db, &query);
    scan(c, "scan/LVQ2-8x4", LvqTwo::new(D, 8, 4).unwrap(),        &db, &query);
}

criterion_group!(benches, bench_scans);
criterion_main!(benches);
