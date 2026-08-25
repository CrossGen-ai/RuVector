//! Criterion micro-benchmarks: sketch encoding + Hamming ranking.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_simhash_prefilter::{FlatPrefilterIndex, SketchFamily, SrpFamily};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nrm = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| nrm.sample(&mut rng)).collect())
        .collect()
}

fn bench_sketch_encode(c: &mut Criterion) {
    let dim = 128;
    let v = synth(1, dim, 1).pop().unwrap();
    let f64b = SrpFamily::<1>::new(dim, 1);
    let f128b = SrpFamily::<2>::new(dim, 1);
    let f256b = SrpFamily::<4>::new(dim, 1);
    c.bench_function("encode_64bit_dim128", |b| {
        b.iter(|| f64b.sketch(black_box(&v)).unwrap())
    });
    c.bench_function("encode_128bit_dim128", |b| {
        b.iter(|| f128b.sketch(black_box(&v)).unwrap())
    });
    c.bench_function("encode_256bit_dim128", |b| {
        b.iter(|| f256b.sketch(black_box(&v)).unwrap())
    });
}

fn bench_end_to_end(c: &mut Criterion) {
    let dim = 128;
    let n = 5_000;
    let vectors = synth(n, dim, 42);
    let queries = synth(64, dim, 43);
    let idx_exact = FlatPrefilterIndex::build(SrpFamily::<1>::new(dim, 1), vectors.clone()).unwrap();
    let idx_128 = FlatPrefilterIndex::build(SrpFamily::<2>::new(dim, 1), vectors.clone()).unwrap();
    let idx_256 = FlatPrefilterIndex::build(SrpFamily::<4>::new(dim, 1), vectors).unwrap();
    let mut q_iter = queries.iter().cycle();
    c.bench_function("exact_scan_n5k_dim128_k10", |b| {
        b.iter(|| {
            let q = q_iter.next().unwrap();
            black_box(idx_exact.search_exact(q, 10));
        })
    });
    c.bench_function("prefilter_128b_mult10_n5k_dim128_k10", |b| {
        b.iter(|| {
            let q = q_iter.next().unwrap();
            black_box(idx_128.search_prefilter(q, 10, 10).unwrap());
        })
    });
    c.bench_function("prefilter_256b_mult5_n5k_dim128_k10", |b| {
        b.iter(|| {
            let q = q_iter.next().unwrap();
            black_box(idx_256.search_prefilter(q, 10, 5).unwrap());
        })
    });
}

criterion_group!(benches, bench_sketch_encode, bench_end_to_end);
criterion_main!(benches);
