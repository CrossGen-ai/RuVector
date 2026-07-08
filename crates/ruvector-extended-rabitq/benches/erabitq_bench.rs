//! Standalone (harness=false) benchmark that measures raw scan throughput
//! for 1/2/4/8-bit codes. No criterion dep — this keeps `cargo bench`
//! runnable without extra crates and produces stable numbers.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use ruvector_extended_rabitq::{ExtendedRabitqIndex, AnnIndex};

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let v: f64 = rng.sample(StandardNormal);
                    v as f32
                })
                .collect()
        })
        .collect()
}

fn bench_bits(bits: u32, n: usize, dim: usize) {
    let data = corpus(n, dim, 1);
    let queries = corpus(200, dim, 2);
    let idx = ExtendedRabitqIndex::build(dim, bits, 42, &data).unwrap();
    // Warmup.
    for q in queries.iter().take(20) {
        let _ = idx.search(q, 10).unwrap();
    }
    let t0 = Instant::now();
    for q in &queries {
        let _ = idx.search(q, 10).unwrap();
    }
    let secs = t0.elapsed().as_secs_f32();
    let qps = queries.len() as f32 / secs;
    let ns_per_scan = secs * 1e9 / (queries.len() as f32 * n as f32);
    println!(
        "bits={bits:>2} n={n:>6} dim={dim:>4} → qps={qps:>10.1}  ns/candidate={ns_per_scan:>7.2}"
    );
}

fn main() {
    for &bits in &[1u32, 2, 4, 8] {
        bench_bits(bits, 10_000, 128);
    }
    for &bits in &[1u32, 2, 4, 8] {
        bench_bits(bits, 50_000, 128);
    }
}
