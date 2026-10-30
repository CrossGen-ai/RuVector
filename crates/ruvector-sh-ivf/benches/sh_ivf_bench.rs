use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_sh_ivf::{gen_dataset, scan_full, scan_sh, scan_two_stage, IvfPq};

fn bench_scanners(c: &mut Criterion) {
    let d = 96;
    let n = 20_000;
    let base = gen_dataset(n, d, 64, 11);
    let queries = gen_dataset(100, d, 64, 22);
    let idx = IvfPq::train(&base, d, 128, 12, 256, 7);

    let nprobe = 16;
    let k = 10;

    c.bench_function("full nprobe=16", |b| {
        b.iter(|| {
            for i in 0..queries.len() / d {
                let r = scan_full(&idx, black_box(&queries[i * d..(i + 1) * d]), nprobe, k);
                black_box(r);
            }
        })
    });

    c.bench_function("two-stage M/2 nprobe=16", |b| {
        b.iter(|| {
            for i in 0..queries.len() / d {
                let r = scan_two_stage(
                    &idx,
                    black_box(&queries[i * d..(i + 1) * d]),
                    nprobe,
                    k,
                    6,
                    10,
                );
                black_box(r);
            }
        })
    });

    c.bench_function("SH M/4-M/2 nprobe=16", |b| {
        b.iter(|| {
            for i in 0..queries.len() / d {
                let r = scan_sh(
                    &idx,
                    black_box(&queries[i * d..(i + 1) * d]),
                    nprobe,
                    k,
                    3,
                    6,
                    40,
                    10,
                );
                black_box(r);
            }
        })
    });
}

criterion_group!(benches, bench_scanners);
criterion_main!(benches);
