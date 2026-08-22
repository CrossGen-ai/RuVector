//! Criterion bench: three variants (flat-f32 baseline, 4-stage RVQ,
//! 8-stage RVQ) on synthetic Gaussian data. Real numbers only.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_rvq::{Rvq, RvqConfig, RvqIndex};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn flat_l2(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = data.len() / d;
    let mut scored: Vec<(u32, f32)> = (0..n).map(|i| {
        let mut s = 0f32;
        for j in 0..d { let e = data[i * d + j] - q[j]; s += e * e; }
        (i as u32, s)
    }).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

fn bench_search(c: &mut Criterion) {
    let d = 128usize;
    let n = 5_000usize;
    let data = synth(n, d, 1);
    let queries = synth(64, d, 2);

    // baseline
    c.bench_function("flat_f32_l2_top10_n5k_d128", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let q = &queries[(i % 64) * d..((i % 64) + 1) * d];
            i = i.wrapping_add(1);
            black_box(flat_l2(&data, d, q, 10));
        });
    });

    for &stages in &[4usize, 8] {
        let cfg = RvqConfig { stages, k: 256, kmeans_iters: 10, seed: 7 };
        let rvq = Rvq::train(&data, n, d, &cfg).unwrap();
        let idx = RvqIndex::build(rvq, &data);
        let name = format!("rvq_l2_top10_n5k_d128_stages{stages}");
        c.bench_function(&name, |b| {
            let mut i = 0usize;
            b.iter(|| {
                let q = &queries[(i % 64) * d..((i % 64) + 1) * d];
                i = i.wrapping_add(1);
                black_box(idx.search_l2(q, 10));
            });
        });
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
