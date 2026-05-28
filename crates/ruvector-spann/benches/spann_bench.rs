//! Criterion micro-benchmarks for SPANN closure variants.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_spann::{FixedMultiAssign, SingleAssign, SpannClosure, SpannIndex};

fn gen_clustered(n: usize, d: usize, n_clusters: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| rng.gen::<f32>() * 10.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..n_clusters)];
            (0..d).map(|i| c[i] + (rng.gen::<f32>() - 0.5) * 2.0 * sigma).collect()
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let n = 10_000;
    let dim = 64;
    let k = 64;
    let nprobe = 4;
    let top_k = 10;

    let data = gen_clustered(n, dim, 32, 1.0, 1);
    let queries = gen_clustered(20, dim, 32, 1.0, 2);

    let (single, _) = SpannIndex::build(data.clone(), k, 10, &SingleAssign, 42);
    let (multi, _) =
        SpannIndex::build(data.clone(), k, 10, &FixedMultiAssign { k: 2 }, 42);
    let (spann, _) = SpannIndex::build(
        data.clone(),
        k,
        10,
        &SpannClosure { epsilon: 0.15, cap: 4 },
        42,
    );

    c.bench_function("search_single", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(single.search(black_box(q), top_k, nprobe));
            }
        })
    });
    c.bench_function("search_fixed_multi_k2", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(multi.search(black_box(q), top_k, nprobe));
            }
        })
    });
    c.bench_function("search_spann_eps0.15", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(spann.search(black_box(q), top_k, nprobe));
            }
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
