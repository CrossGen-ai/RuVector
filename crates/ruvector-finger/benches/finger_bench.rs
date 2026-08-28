//! Criterion micro-benchmark for the per-neighbour scoring cost.
//!
//! Isolates the hot inner loop of each estimator (`handle.score(nid)`) so we
//! can report a fair "cycles per neighbour" number in the research doc.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_finger::estimator::DistanceEstimator;
use ruvector_finger::{Dataset, ExactEstimator, FingerEstimator, JlEstimator, PivotIndex};

fn bench_scoring(c: &mut Criterion) {
    let ds = Dataset::synthetic(5_000, 8, 128, 20260828);
    let idx = PivotIndex::build(&ds, 70, 1);
    let query = ds.queries[0].clone();
    let pivot = idx.entry(&query, 1)[0].0;
    let members = idx.members[pivot as usize].clone();

    let exact = ExactEstimator::new(&idx);
    let jl16 = JlEstimator::build(&idx, 16, 42, "jl-r16");
    let f8 = FingerEstimator::build(&idx, 8, 42, "finger-r8");
    let f16 = FingerEstimator::build(&idx, 16, 42, "finger-r16");

    let mut score = |c: &mut Criterion, name: &str, est: &dyn DistanceEstimator| {
        c.bench_function(name, |b| {
            b.iter(|| {
                let h = est.prepare_query(&query, pivot);
                let mut acc = 0f32;
                for &nid in &members {
                    acc += h.score(nid);
                }
                black_box(acc)
            });
        });
    };

    score(c, "exact/score", &exact);
    score(c, "jl-r16/score", &jl16);
    score(c, "finger-r8/score", &f8);
    score(c, "finger-r16/score", &f16);
}

criterion_group!(benches, bench_scoring);
criterion_main!(benches);
