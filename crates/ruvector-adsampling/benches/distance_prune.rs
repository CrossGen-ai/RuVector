use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use ruvector_adsampling::{
    topk_adsample, topk_full, topk_partial, Policy, Projector,
};

fn gauss(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut r = StdRng::seed_from_u64(seed);
    (0..n * d)
        .map(|_| {
            let v: f64 = r.sample(StandardNormal);
            v as f32
        })
        .collect()
}

fn bench_scan(c: &mut Criterion) {
    let n = 5_000;
    let d = 128;
    let k = 10;
    let data = gauss(n, d, 42);
    let q = gauss(1, d, 777);
    let p = Projector::new(d, 11);
    let mut pdata = vec![0f32; n * d];
    p.project_many(&data, &mut pdata);
    let mut pq = vec![0f32; d];
    p.project(&q, &mut pq);

    let mut g = c.benchmark_group("scan5k_d128_k10");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("full_l2", |b| {
        b.iter(|| topk_full(&data, d, &q, k));
    });
    g.bench_function("partial_l2_step32", |b| {
        b.iter(|| topk_partial(&data, d, &q, k, 32));
    });
    g.bench_function("adsample_eps1_5", |b| {
        let pol = Policy::new(32, 1.5);
        b.iter(|| topk_adsample(&pdata, d, &pq, k, pol));
    });
    g.bench_function("adsample_eps2_1", |b| {
        let pol = Policy::new(32, 2.1);
        b.iter(|| topk_adsample(&pdata, d, &pq, k, pol));
    });
    g.bench_function("adsample_eps2_6", |b| {
        let pol = Policy::new(32, 2.6);
        b.iter(|| topk_adsample(&pdata, d, &pq, k, pol));
    });
    g.finish();
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
