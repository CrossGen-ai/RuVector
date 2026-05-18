//! Criterion benchmark: search latency for BaselineGraph vs RoarGraph.
//!
//! Run with:
//!   cargo bench -p ruvector-roargraph

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ruvector_roargraph::baseline::BaselineGraph;
use ruvector_roargraph::build::BuildParams;
use ruvector_roargraph::dataset::{generate_ood_dataset, DatasetParams};
use ruvector_roargraph::{AnnIndex, RoarGraphIndex};

const N_BASE: usize = 2_000;
const DIM: usize = 32;
const N_CLUSTERS: usize = 4;
const N_TRAIN: usize = 200;
const N_TEST: usize = 50;
const K: usize = 10;
const EF: usize = 50;
const MAX_DEGREE: usize = 16;
const K_TRAIN: usize = 10;
const SEED: u64 = 99;

fn build_indices() -> (BaselineGraph, RoarGraphIndex, Vec<Vec<f32>>) {
    let params = DatasetParams {
        n_base: N_BASE,
        dim: DIM,
        n_clusters: N_CLUSTERS,
        cluster_std: 0.5,
        cluster_range: 4.0,
        ood_shift: 2.5,
        seed: SEED,
    };
    let ds = generate_ood_dataset(&params, N_TRAIN, N_TEST, K);

    let mut baseline = BaselineGraph::new(DIM, MAX_DEGREE);
    baseline.add(&ds.base).unwrap();
    baseline.build(&[]).unwrap();

    let roar_params = BuildParams {
        k_train: K_TRAIN,
        max_degree: MAX_DEGREE,
    };
    let mut roar = RoarGraphIndex::new(DIM, roar_params);
    roar.add(&ds.base).unwrap();
    roar.build(&ds.train_queries).unwrap();

    (baseline, roar, ds.test_queries)
}

fn bench_search(c: &mut Criterion) {
    let (baseline, roar, test_queries) = build_indices();

    let mut group = c.benchmark_group("search_latency");

    for &ef in &[20usize, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("baseline", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &test_queries {
                        baseline.search(q, K, ef).unwrap();
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("roargraph", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &test_queries {
                        roar.search(q, K, ef).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
