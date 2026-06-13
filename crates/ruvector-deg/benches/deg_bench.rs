use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand::prelude::*;
use rand::rngs::StdRng;
use ruvector_deg::{AnnIndex, BruteForce, Deg};

fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect()
}

fn bench_search(c: &mut Criterion) {
    let dim = 64;
    let n = 2_000;
    let k = 10;
    let data = gen(n, dim, 42);
    let queries = gen(100, dim, 7);

    let mut bf = BruteForce::new(dim);
    for v in &data { bf.insert(v.clone()); }
    let mut deg = Deg::new(dim, 24, 64, 64);
    for v in &data { deg.insert(v.clone()); }

    c.bench_function("brute_force_search", |b| {
        b.iter(|| {
            for q in &queries { black_box(bf.search(q, k)); }
        })
    });
    c.bench_function("deg_search", |b| {
        b.iter(|| {
            for q in &queries { black_box(deg.search(q, k)); }
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
