use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_betivf::search::{search, SearchStrategy};
use ruvector_betivf::IvfIndex;

fn mixture(dim: usize, n: usize, blobs: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let nb = Normal::new(0.0, 5.0).unwrap();
    let nx = Normal::new(0.0, 0.5).unwrap();
    let centers: Vec<Vec<f32>> = (0..blobs)
        .map(|_| (0..dim).map(|_| nb.sample(&mut rng) as f32).collect())
        .collect();
    let mut d = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % blobs];
        for j in 0..dim {
            d.push(c[j] + nx.sample(&mut rng) as f32);
        }
    }
    d
}

fn bench(c: &mut Criterion) {
    let dim = 64;
    let n = 20_000;
    let data = mixture(dim, n, 128, 42);
    let mut rng = StdRng::seed_from_u64(7);
    let idx = IvfIndex::build(dim, data, 128, 8, &mut rng).unwrap();
    let queries = mixture(dim, 256, 128, 99);
    let k = 10;

    let mut g = c.benchmark_group("betivf");
    g.bench_function("fixed_nprobe_16", |b| {
        b.iter(|| {
            for qi in 0..256 {
                let q = &queries[qi * dim..(qi + 1) * dim];
                black_box(search(&idx, q, k, SearchStrategy::FixedNprobe(16)));
            }
        });
    });
    g.bench_function("bet_slack_1.0", |b| {
        b.iter(|| {
            for qi in 0..256 {
                let q = &queries[qi * dim..(qi + 1) * dim];
                black_box(search(
                    &idx,
                    q,
                    k,
                    SearchStrategy::BoundedEarlyTerm {
                        max_nprobe: 128,
                        slack: 1.0,
                    },
                ));
            }
        });
    });
    g.bench_function("bet_slack_0.7", |b| {
        b.iter(|| {
            for qi in 0..256 {
                let q = &queries[qi * dim..(qi + 1) * dim];
                black_box(search(
                    &idx,
                    q,
                    k,
                    SearchStrategy::BoundedEarlyTerm {
                        max_nprobe: 128,
                        slack: 0.7,
                    },
                ));
            }
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
