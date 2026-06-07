use criterion::{criterion_group, criterion_main, Criterion};
use ruvector_dedrift::{
    dedrift::{apply, Policy, PolicyConfig},
    drift_sim::DriftWorld,
    Ivf, SmallRng,
};

fn build_pre_drift(dim: usize, n_lists: usize, n_init: usize, n_drift: usize) -> Ivf {
    let world = DriftWorld::new(dim, 8, 7);
    let mut rng = SmallRng::new(11);
    let init = world.batch(n_init, 0.0, &mut rng);
    let mut ivf = Ivf::new(dim, n_lists);
    ivf.train(&init, 6, 91);
    for v in &init {
        ivf.add(v);
    }
    let drift = world.batch(n_drift, 4.0, &mut rng);
    for v in &drift {
        ivf.add(v);
    }
    ivf
}

fn bench_policies(c: &mut Criterion) {
    let dim = 32;
    let n_lists = 32;
    let n_init = 4_000;
    let n_drift = 4_000;
    let cfg = PolicyConfig::default();

    let mut group = c.benchmark_group("dedrift_policies");
    group.sample_size(20);

    for (label, policy) in [
        ("Split", Policy::Split),
        ("Lazy", Policy::Lazy),
        ("Hybrid", Policy::Hybrid),
        ("FullRebuild", Policy::FullRebuild),
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || build_pre_drift(dim, n_lists, n_init, n_drift),
                |mut ivf| {
                    let _ = apply(&mut ivf, policy, &cfg);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_policies);
criterion_main!(benches);
