use criterion::{Criterion, criterion_group, criterion_main};
use ruvector_adaptive_probe::dataset::Workload;
use ruvector_adaptive_probe::{
    FixedNprobe, IvfIndex, MarginBudget, PlateauProbe, ProbeStrategy,
};
use std::hint::black_box;

fn build_workload() -> (IvfIndex, Workload) {
    let n = 10_000usize;
    let dim = 64usize;
    let n_queries = 200usize;
    let wl = Workload::gaussian(n, n_queries, dim, 24, 0xbeef);
    let idx = IvfIndex::build(&wl.corpus, n, dim, 48, 13).unwrap();
    (idx, wl)
}

fn run_search<S: ProbeStrategy>(idx: &IvfIndex, wl: &Workload, strat: &S) {
    for qi in 0..wl.n_queries {
        let q = wl.query(qi);
        let r = idx.search(q, 10, 16, strat).unwrap();
        black_box(r);
    }
}

fn bench_strategies(c: &mut Criterion) {
    let (idx, wl) = build_workload();
    let mut g = c.benchmark_group("ivf_search_200q_k10");
    g.bench_function("fixed_nprobe_16", |b| {
        let s = FixedNprobe::new(16);
        b.iter(|| run_search(&idx, &wl, &s));
    });
    g.bench_function("fixed_nprobe_4", |b| {
        let s = FixedNprobe::new(4);
        b.iter(|| run_search(&idx, &wl, &s));
    });
    g.bench_function("plateau_patience_2", |b| {
        let s = PlateauProbe::new(2);
        b.iter(|| run_search(&idx, &wl, &s));
    });
    g.bench_function("margin_m6_w2", |b| {
        let s = MarginBudget::new(6.0, 2);
        b.iter(|| run_search(&idx, &wl, &s));
    });
    g.finish();
}

criterion_group!(benches, bench_strategies);
criterion_main!(benches);
