//! Micro-benchmarks for the per-query inner loop.
//!
//! No `criterion` dependency — this is a hand-rolled wall-clock
//! micro-benchmark that reports mean & p99 latency for a fixed query
//! stream. Keeps the crate dependency-light.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_rabitq_ensemble::{Index, RabitqEnsemble, RabitqSingle, RotationKind};

fn gen(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0_f32..=1.0)).collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 0.0 {
                let inv = nrm.recip();
                for x in &mut v {
                    *x *= inv;
                }
            }
            v
        })
        .collect()
}

fn stats(mut xs: Vec<f32>) -> (f32, f32) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f32 = xs.iter().copied().sum::<f32>() / xs.len() as f32;
    let p99_idx = ((xs.len() as f32) * 0.99) as usize;
    (mean, xs[p99_idx.min(xs.len() - 1)])
}

fn bench<I: Index>(name: &str, idx: &I, queries: &[Vec<f32>]) {
    // Warm up — one pass over queries.
    for q in queries.iter().take(32) {
        let _ = idx.search(q, 10);
    }
    let mut per_query: Vec<f32> = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = Instant::now();
        let _r = idx.search(q, 10);
        per_query.push(t0.elapsed().as_micros() as f32);
    }
    let (mean, p99) = stats(per_query);
    println!("  {:<28}  mean={:>7.1}µs  p99={:>7.1}µs", name, mean, p99);
}

fn sweep(n: usize, d: usize, nq: usize, seed: u64) {
    println!("\n== n={n} d={d} nq={nq} ==");
    let rows = gen(n, d, seed);
    let queries = gen(nq, d, seed ^ 0x1234);
    let kind = RotationKind::HadamardSigned;
    let single = RabitqSingle::build(&rows, kind, seed);
    bench("single", &single, &queries);
    for &k in &[2usize, 4, 8] {
        let e = RabitqEnsemble::build(&rows, k, kind, seed);
        bench(&format!("ensemble K={k}"), &e, &queries);
    }
}

fn main() {
    sweep(10_000, 128, 100, 0xA1);
    sweep(50_000, 128, 100, 0xB2);
    sweep(100_000, 128, 50, 0xC3);
}
