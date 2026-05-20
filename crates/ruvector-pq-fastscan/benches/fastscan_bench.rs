//! Criterion bench: scan-only throughput for the three variants on a 50k×128 set.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_pq_fastscan::{flat_l2_topk, FastScanIndex, Pq8Index, ProductQuantizer};

fn gen(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0f32, 1f32).unwrap();
    let cents: Vec<f32> = (0..32 * d).map(|_| rng.gen_range(-4.0..4.0)).collect();
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        let c = rng.gen_range(0..32);
        for j in 0..d {
            out[i * d + j] = cents[c * d + j] + normal.sample(&mut rng);
        }
    }
    out
}

fn bench_scan(c: &mut Criterion) {
    let n = 50_000usize;
    let d = 128usize;
    let m = 16usize;
    let train_n = 10_000usize;

    let data = gen(n, d, 1);
    let queries = gen(64, d, 2);
    let train = &data[..train_n * d];

    let pq8 = ProductQuantizer::train(train, train_n, d, m, 256, 8, 1).unwrap();
    let pq8_idx = Pq8Index::from_vectors(pq8, &data, n);
    let fs = FastScanIndex::from_vectors(train, train_n, &data, n, d, m, 8, 2).unwrap();

    c.bench_function("flat_l2_topk_50k", |b| {
        b.iter(|| {
            let q = &queries[0..d];
            black_box(flat_l2_topk(&data, n, d, q, 10))
        })
    });

    c.bench_function("pq8_search_50k", |b| {
        b.iter(|| {
            let q = &queries[0..d];
            black_box(pq8_idx.search(q, 10))
        })
    });

    c.bench_function("fastscan4_search_50k", |b| {
        b.iter(|| {
            let q = &queries[0..d];
            let lut = fs.build_lut(q);
            black_box(fs.search_u16(&lut, 10))
        })
    });
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
