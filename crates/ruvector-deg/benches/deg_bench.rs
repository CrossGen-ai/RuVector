use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_deg::distance::Metric;
use ruvector_deg::graph::{DegGraph, DegParams};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()).collect()
}

fn build_bench(c: &mut Criterion) {
    let dim = 64;
    let mut group = c.benchmark_group("deg_build");
    for n in [1000usize, 2000, 4000] {
        let data = random_vectors(n, dim, 1);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                let params = DegParams { degree: 24, eps: 60, refine: 4, metric: Metric::L2Sq, seed: 1 };
                let mut g = DegGraph::new(dim, params);
                for v in &data { g.insert(v); }
                g.len()
            });
        });
    }
    group.finish();
}

fn query_bench(c: &mut Criterion) {
    let dim = 64; let n = 4000; let k = 10;
    let data = random_vectors(n, dim, 1);
    let queries = random_vectors(100, dim, 2);
    let params = DegParams { degree: 24, eps: 60, refine: 4, metric: Metric::L2Sq, seed: 1 };
    let mut g = DegGraph::new(dim, params);
    for v in &data { g.insert(v); }

    let mut group = c.benchmark_group("deg_query");
    for eps in [30usize, 60, 120] {
        let p = DegParams { eps, ..g.params().clone() };
        // We can't change params after the fact in this minimal API, so
        // simulate by tuning k vs the default eps. Instead, just sweep k.
        let _ = p;
        group.bench_function(BenchmarkId::from_parameter(eps), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for q in &queries {
                    total += g.search(q, k.min(eps)).len();
                }
                total
            });
        });
    }
    group.finish();
}

fn churn_bench(c: &mut Criterion) {
    let dim = 64; let n = 2000;
    let data = random_vectors(n, dim, 1);
    let fresh = random_vectors(500, dim, 2);
    c.bench_function("deg_churn_delete_500_insert_500", |b| {
        b.iter(|| {
            let params = DegParams { degree: 24, eps: 60, refine: 4, metric: Metric::L2Sq, seed: 1 };
            let mut g = DegGraph::new(dim, params);
            for v in &data { g.insert(v); }
            for id in 0..500u32 { g.delete(id); }
            for v in &fresh { g.insert(v); }
            g.len()
        });
    });
}

criterion_group!(benches, build_bench, query_bench, churn_bench);
criterion_main!(benches);
