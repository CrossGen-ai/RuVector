use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{AnnIndex, FlatIndex, PqRerankIndex, SymphonyQgIndex};

fn gauss(n: usize, d: usize, kc: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..kc).map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = &centers[i % kc];
        for j in 0..d { out.push(c[j] + normal.sample(&mut rng) as f32); }
    }
    out
}

fn bench_search(c: &mut Criterion) {
    let n = 4_000; let d = 64; let k = 10;
    let data = gauss(n, d, 16, 1.0, 1);
    let qs = gauss(64, d, 16, 1.0, 2);

    let flat = FlatIndex::new(d, data.clone());
    let pqr = PqRerankIndex::build(d, data.clone(), 8, 64, 10, 100, 7);
    let sym = SymphonyQgIndex::build(d, data, 8, 64, 10, 16, 64, 1.2, 7).with_ef_search(64);

    c.bench_function("flat_search", |b| b.iter(|| {
        for i in 0..qs.len()/d { let _ = black_box(flat.search(&qs[i*d..(i+1)*d], k)); }
    }));
    c.bench_function("pq_rerank_search", |b| b.iter(|| {
        for i in 0..qs.len()/d { let _ = black_box(pqr.search(&qs[i*d..(i+1)*d], k)); }
    }));
    c.bench_function("symphony_qg_search", |b| b.iter(|| {
        for i in 0..qs.len()/d { let _ = black_box(sym.search(&qs[i*d..(i+1)*d], k)); }
    }));
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
