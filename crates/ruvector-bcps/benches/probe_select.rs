use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_bcps::{binary::BinarySignatureSelector, exact::ExactL2Selector, ProbeSelector};

fn gen(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
}

fn bench_select(c: &mut Criterion) {
    let d = 128usize;
    let nprobe = 16usize;
    let qs = gen(32, d, 2);

    let mut group = c.benchmark_group("probe_select");
    for &nlist in &[4096usize, 16384, 65536] {
        let cs = gen(nlist, d, 1);
        let exact = ExactL2Selector::new(cs.clone());
        group.bench_with_input(BenchmarkId::new("exact", nlist), &nlist, |b, _| {
            b.iter(|| {
                for q in &qs { std::hint::black_box(exact.select(q, nprobe)); }
            });
        });
        for &m in &[nprobe * 8, nprobe * 16] {
            let bin = BinarySignatureSelector::new(cs.clone(), m, 42);
            group.bench_with_input(
                BenchmarkId::new(format!("bcps_m{}", m), nlist),
                &nlist,
                |b, _| {
                    b.iter(|| {
                        for q in &qs { std::hint::black_box(bin.select(q, nprobe)); }
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
