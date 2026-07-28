//! Criterion bench that runs the three oracle variants over a fixed
//! synthetic corpus. Numbers are real — captured into
//! docs/research/nightly/2026-07-28-adsampling-early-termination/README.md.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use ruvector_adsampling::{bench_variants, VariantReport};

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| r.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

fn bench_all(c: &mut Criterion) {
    // Moderate corpus/query sizes so the bench finishes in < ~30s.
    let corpus = synth(8_192, 128, 0xABCD);
    let queries = synth(64, 128, 0x1234);
    let k = 10;
    let mut g = c.benchmark_group("adsampling-variants");
    g.sample_size(10);
    g.bench_function("full-workload", |b| {
        b.iter(|| {
            let rows: Vec<VariantReport> = bench_variants(&corpus, &queries, k, 42);
            // Sanity so Criterion cannot dead-code us:
            assert_eq!(rows.len(), 3);
        });
    });
    g.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
