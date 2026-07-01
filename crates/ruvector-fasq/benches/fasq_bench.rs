use criterion::{criterion_group, criterion_main, black_box, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_fasq::{baseline::{UniformSq4, UniformSq8}, quantizer::Fasq, Quantizer};

fn make(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let stds: Vec<f32> = (0..d).map(|i| 4.0 * 0.85f32.powi(i as i32) + 0.05).collect();
    let normals: Vec<Normal<f32>> = stds.iter().map(|s| Normal::new(0.0, *s).unwrap()).collect();
    (0..n).map(|_| (0..d).map(|i| normals[i].sample(&mut rng)).collect()).collect()
}

fn bench_encode(c: &mut Criterion) {
    let d = 128;
    let train = make(2000, d, 1);
    let queries = make(1000, d, 2);
    let sq8 = UniformSq8::train(&train).unwrap();
    let sq4 = UniformSq4::train(&train).unwrap();
    let fasq = Fasq::train(&train, 4.0, 2, 8).unwrap();

    let mut g = c.benchmark_group("encode_1k_d128");
    g.bench_function("sq8", |b| b.iter(|| {
        let mut out = Vec::with_capacity(1024);
        for v in &queries {
            out.clear();
            sq8.encode(black_box(v), &mut out).unwrap();
        }
    }));
    g.bench_function("sq4", |b| b.iter(|| {
        let mut out = Vec::with_capacity(1024);
        for v in &queries {
            out.clear();
            sq4.encode(black_box(v), &mut out).unwrap();
        }
    }));
    g.bench_function("fasq_avg4", |b| b.iter(|| {
        let mut out = Vec::with_capacity(1024);
        for v in &queries {
            out.clear();
            fasq.encode(black_box(v), &mut out).unwrap();
        }
    }));
    g.finish();
}

fn bench_distance(c: &mut Criterion) {
    let d = 128;
    let train = make(2000, d, 3);
    let base = make(1000, d, 4);
    let query = make(1, d, 5).into_iter().next().unwrap();
    let sq8 = UniformSq8::train(&train).unwrap();
    let sq4 = UniformSq4::train(&train).unwrap();
    let fasq = Fasq::train(&train, 4.0, 2, 8).unwrap();

    let mut codes_sq8 = Vec::new();
    let mut codes_sq4 = Vec::new();
    let mut codes_fasq = Vec::new();
    for v in &base {
        let mut c = Vec::with_capacity(64);
        sq8.encode(v, &mut c).unwrap(); codes_sq8.push(c);
        let mut c = Vec::with_capacity(64);
        sq4.encode(v, &mut c).unwrap(); codes_sq4.push(c);
        let mut c = Vec::with_capacity(64);
        fasq.encode(v, &mut c).unwrap(); codes_fasq.push(c);
    }

    let mut g = c.benchmark_group("distance_1k_d128");
    g.bench_function("sq8", |b| b.iter(|| {
        let mut s = Vec::new();
        let mut acc = 0.0f32;
        for c in &codes_sq8 {
            acc += sq8.distance_sq(black_box(&query), c, &mut s).unwrap();
        }
        black_box(acc)
    }));
    g.bench_function("sq4", |b| b.iter(|| {
        let mut s = Vec::new();
        let mut acc = 0.0f32;
        for c in &codes_sq4 {
            acc += sq4.distance_sq(black_box(&query), c, &mut s).unwrap();
        }
        black_box(acc)
    }));
    g.bench_function("fasq_avg4", |b| b.iter(|| {
        let mut s = Vec::new();
        let mut acc = 0.0f32;
        for c in &codes_fasq {
            acc += fasq.distance_sq(black_box(&query), c, &mut s).unwrap();
        }
        black_box(acc)
    }));
    g.finish();
}

criterion_group!(benches, bench_encode, bench_distance);
criterion_main!(benches);
