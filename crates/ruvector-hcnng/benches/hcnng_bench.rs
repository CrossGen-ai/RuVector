//! Criterion microbenchmarks for HCNNG build and search.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use ruvector_hcnng::{HcnngIndex, HcnngParams, Metric};

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect())
        .collect()
}

fn bench_build(c: &mut Criterion) {
    let data = synth(5_000, 64, 1);
    c.bench_function("hcnng_build_n5k_d64_t10", |b| {
        b.iter(|| {
            let p = HcnngParams {
                n_trees: 10,
                leaf_size: 32,
                max_degree: 32,
                knn_per_node: 3,
                ef_search: 64,
                seed: 0,
                metric: Metric::L2Sq,
            };
            let idx = HcnngIndex::build(black_box(data.clone()), p).unwrap();
            black_box(idx);
        })
    });
}

fn bench_search(c: &mut Criterion) {
    let data = synth(20_000, 64, 1);
    let q = synth(1, 64, 2).pop().unwrap();
    let p = HcnngParams {
        n_trees: 12,
        leaf_size: 32,
        max_degree: 32,
        knn_per_node: 3,
        ef_search: 64,
        seed: 0,
        metric: Metric::L2Sq,
    };
    let idx = HcnngIndex::build(data, p).unwrap();
    c.bench_function("hcnng_search_n20k_d64_ef64_k10", |b| {
        b.iter(|| {
            let r = idx.search(black_box(&q), 10).unwrap();
            black_box(r);
        })
    });
}

criterion_group!(benches, bench_build, bench_search);
criterion_main!(benches);
