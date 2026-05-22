use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_symphonyqg::{SymphonyQg, SymphonyQgParams};
use ruvector_symphonyqg::index::brute_force;

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect()).collect()
}

fn bench_search(c: &mut Criterion) {
    let n = 5_000;
    let d = 128;
    let k = 10;
    let db = synth(n, d, 42);
    let queries = synth(64, d, 99);
    let idx = SymphonyQg::build(db.clone(), SymphonyQgParams {
        m: 16, ef_construction: 64, ef_search: 64, rotation_seed: 1,
    });

    let mut group = c.benchmark_group("symphonyqg_n5k_d128");
    group.bench_function("brute_force", |b| {
        b.iter(|| { for q in &queries { let _ = brute_force(&db, q, k); } });
    });
    group.bench_function("nsw_exact_graph", |b| {
        b.iter(|| { for q in &queries { let _ = idx.search_exact_graph(q, k); } });
    });
    group.bench_function("symphonyqg", |b| {
        b.iter(|| { for q in &queries { let _ = idx.search(q, k); } });
    });
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
