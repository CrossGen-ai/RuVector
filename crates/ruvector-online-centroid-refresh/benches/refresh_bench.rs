//! Standalone benchmark (no criterion dep) — prints csv-style summary.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_online_centroid_refresh::{IvfConfig, OnlineIvf, RefreshPolicy, Vector};
use std::time::Instant;

fn gen(seed: u64, n: usize, dim: usize, c: f32, s: f32) -> Vec<Vector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| c + (rng.gen::<f32>() - 0.5) * s)
                .collect()
        })
        .collect()
}

fn run(name: &str, dim: usize, k: usize, policy: RefreshPolicy) {
    let train = gen(1, 2_000, dim, 0.0, 1.0);
    let mut stream: Vec<Vector> = Vec::new();
    for i in 0..6 {
        stream.extend(gen(100 + i, 5_000, dim, (i as f32) * 1.5, 1.0));
    }
    let eval = {
        let mut e = Vec::new();
        for i in 0..6 {
            e.extend(gen(200 + i, 500, dim, (i as f32) * 1.5, 1.0));
        }
        e
    };
    let mut cfg = IvfConfig::new(dim, k);
    cfg.policy = policy;
    cfg.reservoir_capacity = 1_500;
    cfg.refresh_iters = 6;
    let mut idx = OnlineIvf::new(cfg, &train);
    let t = Instant::now();
    for v in &stream {
        idx.insert(v);
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;
    let inertia = idx.inertia(&eval);
    let per_insert_us = total_ms * 1e3 / stream.len() as f64;
    println!(
        "{},{},{},{:.3},{:.2},{:.1},{}",
        name, dim, k, per_insert_us, total_ms, inertia, idx.refresh_count,
    );
}

fn main() {
    println!("policy,dim,k,us_per_insert,total_ms,final_inertia,refreshes");
    for &(name, policy) in &[
        ("static", RefreshPolicy::Static),
        ("periodic5k", RefreshPolicy::Periodic(5_000)),
        ("drift015", RefreshPolicy::DriftTriggered(0.15)),
    ] {
        for &(dim, k) in &[(16, 8), (32, 16), (64, 32)] {
            run(name, dim, k, policy);
        }
    }
}
