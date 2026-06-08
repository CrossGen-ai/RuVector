//! Criterion microbenchmarks for ruvector-deg.
//!
//! Reports three variants:
//!   * baseline: edges_per_node=16, eps_insert=40
//!   * balanced: edges_per_node=24, eps_insert=80   (default)
//!   * recall:   edges_per_node=32, eps_insert=160
//!
//! Captured numbers live in docs/research/nightly/<date>-dynamic-exploration-graph/.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_deg::{Deg, DegConfig, L2};

fn random(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..(n * d) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s & 0xFFFFFF) as f32 / 16_777_216.0;
        out.push(u * 2.0 - 1.0);
    }
    out
}

fn variants() -> [(&'static str, DegConfig); 3] {
    [
        ("baseline", DegConfig { dim: 64, edges_per_node: 16, eps_insert: 40 }),
        ("balanced", DegConfig { dim: 64, edges_per_node: 24, eps_insert: 80 }),
        ("recall", DegConfig { dim: 64, edges_per_node: 32, eps_insert: 160 }),
    ]
}

fn bench_build(c: &mut Criterion) {
    let d = 64;
    let n = 1_000;
    let vecs = random(n, d, 0xABCD);
    let mut g = c.benchmark_group("build/1000");
    for (name, cfg) in variants() {
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut deg = Deg::new(cfg);
                deg.build::<L2>(black_box(&vecs));
                black_box(deg.len())
            })
        });
    }
    g.finish();
}

fn bench_query(c: &mut Criterion) {
    let d = 64;
    let n = 2_000;
    let vecs = random(n, d, 0xABCD);
    let queries = random(100, d, 0x1234);
    let mut g = c.benchmark_group("query/2000/k10");
    for (name, cfg) in variants() {
        let mut deg = Deg::new(cfg);
        deg.build::<L2>(&vecs);
        g.bench_function(name, |b| {
            b.iter(|| {
                let mut hits = 0usize;
                for i in 0..100 {
                    let q = &queries[i * d..(i + 1) * d];
                    hits += deg.query::<L2>(black_box(q), 10, 80).len();
                }
                black_box(hits)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_build, bench_query);
criterion_main!(benches);
