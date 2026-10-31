//! Criterion benchmarks: PVQ encode + scan vs float32 flat.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use ruvector_spherical_pvq::{l2_normalize, PvqCodec, PvqIndex};

fn random_unit(d: usize, rng: &mut ChaCha12Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
    l2_normalize(&mut v);
    v
}

fn bench(c: &mut Criterion) {
    let d = 128;
    let n = 10_000;
    let mut rng = ChaCha12Rng::seed_from_u64(0xBEEF);
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_unit(d, &mut rng)).collect();
    let q = random_unit(d, &mut rng);

    for &k in &[16_u32, 32, 64] {
        let codec = PvqCodec::new(d, k);
        let index = PvqIndex::build(codec.clone(), &vectors);

        c.bench_function(&format!("pvq_encode_K{k}"), |b| {
            b.iter(|| black_box(codec.encode(black_box(&q))));
        });

        c.bench_function(&format!("pvq_scan_K{k}_10k"), |b| {
            b.iter(|| black_box(index.search(black_box(&q), 10)));
        });
    }

    // Flat f32 scan baseline (same n×d).
    c.bench_function("flat_f32_scan_10k", |b| {
        b.iter(|| {
            let mut best = (0_usize, f32::MIN);
            for (i, v) in vectors.iter().enumerate() {
                let s: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                if s > best.1 {
                    best = (i, s);
                }
            }
            black_box(best)
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
