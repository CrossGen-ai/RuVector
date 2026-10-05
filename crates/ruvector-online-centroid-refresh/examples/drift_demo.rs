//! Compare Static / Periodic / DriftTriggered policies on a shifting stream.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_online_centroid_refresh::{IvfConfig, OnlineIvf, RefreshPolicy, Vector};
use std::time::Instant;

fn gen_cluster(seed: u64, n: usize, dim: usize, center: f32, spread: f32) -> Vec<Vector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| center + (rng.gen::<f32>() - 0.5) * spread)
                .collect()
        })
        .collect()
}

fn main() {
    let dim = 32;
    let k = 16;
    let train = gen_cluster(1, 2_000, dim, 0.0, 1.0);
    // Simulate a 3-regime stream: steady, drifting mean, new mode appears.
    let mut stream: Vec<Vector> = Vec::new();
    stream.extend(gen_cluster(2, 10_000, dim, 0.0, 1.0));
    stream.extend(gen_cluster(3, 10_000, dim, 3.0, 1.0));
    stream.extend(gen_cluster(4, 10_000, dim, 6.0, 1.5));
    let eval = {
        let mut e = gen_cluster(5, 2_000, dim, 0.0, 1.0);
        e.extend(gen_cluster(6, 2_000, dim, 3.0, 1.0));
        e.extend(gen_cluster(7, 2_000, dim, 6.0, 1.5));
        e
    };

    let policies: Vec<(&str, RefreshPolicy)> = vec![
        ("static", RefreshPolicy::Static),
        ("periodic/5000", RefreshPolicy::Periodic(5_000)),
        ("drift/0.15", RefreshPolicy::DriftTriggered(0.15)),
    ];

    println!(
        "{:>14} | {:>10} | {:>10} | {:>12} | {:>10}",
        "policy", "refreshes", "ms", "final_inertia", "mean_r"
    );
    println!("{}", "-".repeat(70));

    for (name, policy) in policies {
        let mut cfg = IvfConfig::new(dim, k);
        cfg.policy = policy;
        cfg.reservoir_capacity = 1_000;
        cfg.refresh_iters = 8;
        let mut idx = OnlineIvf::new(cfg, &train);
        let t = Instant::now();
        for v in &stream {
            idx.insert(v);
        }
        let elapsed = t.elapsed().as_millis();
        let inertia = idx.inertia(&eval);
        let n_eval = eval.len() as f64;
        println!(
            "{:>14} | {:>10} | {:>10} | {:>12.2} | {:>10.4}",
            name,
            idx.refresh_count,
            elapsed,
            inertia,
            (inertia / n_eval).sqrt()
        );
    }
}
