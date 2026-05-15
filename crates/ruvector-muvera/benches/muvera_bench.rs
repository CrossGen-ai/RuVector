use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_muvera::{chamfer_similarity, FdeConfig, FdeEncoder, FillStrategy, ProjectionMode};

fn unit_multi(n_tokens: usize, d: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    (0..n_tokens)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(rng)).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in v.iter_mut() {
                *x /= n;
            }
            v
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn bench(c: &mut Criterion) {
    let d = 64;
    let q_tokens = 16;
    let d_tokens = 32;
    let mut rng = StdRng::seed_from_u64(7);
    let query = unit_multi(q_tokens, d, &mut rng);
    let doc = unit_multi(d_tokens, d, &mut rng);

    let cfg = FdeConfig {
        d, k_sim: 5, r_reps: 20,
        fill: FillStrategy::NearestBucket,
        projection: ProjectionMode::None, d_final: 0, seed: 42,
    };
    let enc = FdeEncoder::new(cfg).unwrap();
    let q_fde = enc.encode_query(&query).unwrap();
    let d_fde = enc.encode_doc(&doc).unwrap();

    c.bench_function("chamfer_q16_d32_d64", |b| {
        b.iter(|| chamfer_similarity(&query, &doc))
    });
    c.bench_function("fde_dot_k5_R20_d64", |b| {
        b.iter(|| dot(&q_fde, &d_fde))
    });
    c.bench_function("fde_encode_doc", |b| {
        b.iter_batched(|| doc.clone(), |d| enc.encode_doc(&d).unwrap(), BatchSize::SmallInput)
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
