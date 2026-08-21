use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use ruvector_hamming_cascade::{
    Cascade, CascadeConfig, Fp32Oracle, HammingOracle, Int8Oracle,
};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn bench_cascade(c: &mut Criterion) {
    let n = 5_000;
    let dim = 128;
    let db = synth(n, dim, 0xC0FFEE);
    let q: Vec<f32> = synth(1, dim, 0xBEEF).pop().unwrap();

    let mut group = c.benchmark_group("cascade");

    let mut fp32 = Cascade::new(
        Fp32Oracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k: 10, probe_k: 10 },
    );
    group.bench_function(BenchmarkId::new("fp32", n), |b| {
        b.iter(|| fp32.search(&q))
    });

    let mut int8 = Cascade::new(
        Int8Oracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k: 10, probe_k: 100 },
    );
    group.bench_function(BenchmarkId::new("int8_rerank", n), |b| {
        b.iter(|| int8.search(&q))
    });

    let mut ham = Cascade::new(
        HammingOracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k: 10, probe_k: 100 },
    );
    group.bench_function(BenchmarkId::new("hamming_rerank", n), |b| {
        b.iter(|| ham.search(&q))
    });

    group.finish();
}

criterion_group!(benches, bench_cascade);
criterion_main!(benches);
