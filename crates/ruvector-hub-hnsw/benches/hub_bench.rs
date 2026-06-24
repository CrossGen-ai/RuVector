//! Criterion micro-benchmark: per-query search cost for each NSW variant.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_hub_hnsw::{AnnIndex, BaselineNsw, HubNsw, IndegreeCap, NswParams, Vector};

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vector> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
        .collect()
}

fn bench_variants(c: &mut Criterion) {
    let n = 2_000;
    let d = 64;
    let vs = synth(n, d, 1);
    let queries = synth(64, d, 2);
    let p = NswParams {
        m: 16,
        ef_construction: 64,
        seed_entry: 0,
    };
    let base = BaselineNsw::build(vs.clone(), p.clone());
    let light = HubNsw::build(vs.clone(), p.clone(), IndegreeCap::Light);
    let agg = HubNsw::build(vs, p, IndegreeCap::Aggressive);

    let mut g = c.benchmark_group("nsw_search_k10_ef64");
    g.bench_function("baseline", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(base.search(q, 10, 64));
            }
        })
    });
    g.bench_function("light", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(light.search(q, 10, 64));
            }
        })
    });
    g.bench_function("aggressive", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(agg.search(q, 10, 64));
            }
        })
    });
    g.finish();
}

criterion_group!(benches, bench_variants);
criterion_main!(benches);
