//! Criterion bench: brute-force vs NN-Descent variants at fixed N/D/K.
//!
//!     cargo bench -p ruvector-nndescent

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvector_nndescent::{
    brute::BruteForce, nndescent::{NnDescent, NnDescentConfig},
    KnnGraphBuilder, L2,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_graph_build");
    group.sample_size(10);
    for &n in &[1000usize, 2000] {
        let data = gaussian(n, 64, 42);
        let k = 20;
        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| BruteForce::new(L2).build(&data, k));
        });
        group.bench_with_input(BenchmarkId::new("nnd-rho1.0", n), &n, |b, _| {
            b.iter(|| NnDescent::new(L2, NnDescentConfig {
                rho: 1.0, reverse: true, ..Default::default()
            }).build(&data, k));
        });
        group.bench_with_input(BenchmarkId::new("nnd-rho0.5", n), &n, |b, _| {
            b.iter(|| NnDescent::new(L2, NnDescentConfig {
                rho: 0.5, reverse: true, ..Default::default()
            }).build(&data, k));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
