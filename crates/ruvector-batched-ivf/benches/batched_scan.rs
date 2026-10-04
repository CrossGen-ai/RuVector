use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use ruvector_batched_ivf::{
    batched_list_major_scan, batched_tiled_scan, build_ivf, serial_scan, QueryBatch,
};

fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

fn bench_variants(c: &mut Criterion) {
    let dim = 128;
    let n = 20_000;
    let n_lists = 128;
    let n_probe = 12;
    let k = 10;
    let vectors = gen(n, dim, 1);
    let index = build_ivf(vectors, dim, n_lists, 3, 2);

    for &bs in &[1usize, 8, 32, 128] {
        let queries = gen(bs, dim, 7);
        let batch = QueryBatch::new(queries, dim, k, n_probe);
        let probes = batch.probe(&index);

        c.bench_function(&format!("serial_bs{bs}"), |b| {
            b.iter(|| black_box(serial_scan(&index, &batch, &probes)))
        });
        c.bench_function(&format!("list_major_bs{bs}"), |b| {
            b.iter(|| black_box(batched_list_major_scan(&index, &batch, &probes)))
        });
        c.bench_function(&format!("tiled_bs{bs}"), |b| {
            b.iter(|| black_box(batched_tiled_scan(&index, &batch, &probes)))
        });
    }
}

criterion_group!(benches, bench_variants);
criterion_main!(benches);
