use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_nsg::{brute_force_topk, NsgBuilder, NsgParams};

fn gauss_clusters(n: usize, d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..c)
        .map(|_| (0..d).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let cc = &centers[rng.gen_range(0..c)];
            cc.iter().map(|x| x + rng.gen_range(-0.5..0.5)).collect()
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let data = gauss_clusters(5_000, 32, 16, 1);
    let queries = gauss_clusters(100, 32, 16, 2);

    let index = NsgBuilder::new(NsgParams {
        r: 32, l_build: 64, k_knn: 50, knn_iters: 6, knn_sample: 1.0, seed: 7, alpha: 1.2,
    })
    .build(data.clone())
    .unwrap();

    c.bench_function("nsg_search_k10_L64", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for q in &queries {
                let r = index.search(black_box(q), 10, 64).unwrap();
                acc += r.len() as u64;
            }
            black_box(acc);
        });
    });

    c.bench_function("brute_force_k10", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for q in &queries {
                let r = brute_force_topk(black_box(&data), black_box(q), 10);
                acc += r.len() as u64;
            }
            black_box(acc);
        });
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
