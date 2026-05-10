use criterion::{criterion_group, criterion_main, Criterion};
use ruvector_finger::bench_harness::{make_gaussian, BenchConfig};
use ruvector_finger::*;

fn bench_distance_estimators(c: &mut Criterion) {
    let cfg = BenchConfig {
        n: 2_000,
        d: 128,
        q: 1,
        k: 10,
        seed_data: 1,
        seed_query: 2,
    };
    let base = make_gaussian(cfg.n, cfg.d, cfg.seed_data);
    let queries = make_gaussian(8, cfg.d, cfg.seed_query);

    let exact = ExactL2::from_vectors(&base).unwrap();
    let jl32 = JlProjector::new(&base, 32, 7).unwrap();
    let finger32 = FingerEstimator::new(&base, 32, 0.0, 7).unwrap();

    c.bench_function("exact-fp32-topk", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = exact_top_k(&exact, q, cfg.k);
            }
        })
    });

    c.bench_function("jl32-topk", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = jl_top_k(&jl32, q, cfg.k);
            }
        })
    });

    c.bench_function("finger32-rerank50-topk", |b| {
        b.iter(|| {
            for q in &queries {
                let _ = finger_top_k(&finger32, q, cfg.k, 50);
            }
        })
    });
}

criterion_group!(benches, bench_distance_estimators);
criterion_main!(benches);
