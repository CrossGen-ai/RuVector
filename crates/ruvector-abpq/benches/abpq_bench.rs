use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_abpq::{AbPq, AbPqConfig};

fn gen(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scales: Vec<f32> = (0..d).map(|j| 1.0 / (1.0 + j as f32 / 6.0)).collect();
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let normal = Normal::new(0.0, scales[j]).unwrap();
            out[i * d + j] = normal.sample(&mut rng) as f32;
        }
    }
    out
}

fn bench_encode(c: &mut Criterion) {
    let (n, d, m) = (2000, 64, 16);
    let data = gen(n, d, 1);
    let cfg = AbPqConfig::new(d, m, 4);
    let pq = AbPq::train(cfg, &data).unwrap();
    c.bench_function("abpq_encode_batch_2000x64", |b| {
        b.iter(|| black_box(pq.encode_batch(&data)))
    });
}

fn bench_adc_scan(c: &mut Criterion) {
    let (n, d, m) = (10_000, 64, 16);
    let data = gen(n, d, 2);
    let cfg = AbPqConfig::new(d, m, 4);
    let pq = AbPq::train(cfg, &data).unwrap();
    let codes = pq.encode_batch(&data);
    let cb = pq.code_bytes();
    let q = gen(1, d, 3);
    let tbl = pq.build_adc_table(&q);
    c.bench_function("abpq_adc_scan_10000x64", |b| {
        b.iter(|| {
            let mut s = 0f32;
            for i in 0..n {
                s += pq.adc_distance(&codes[i * cb..(i + 1) * cb], &tbl);
            }
            black_box(s)
        })
    });
}

criterion_group!(benches, bench_encode, bench_adc_scan);
criterion_main!(benches);
