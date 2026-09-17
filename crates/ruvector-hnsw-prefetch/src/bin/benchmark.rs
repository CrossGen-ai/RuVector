//! Runnable benchmark for the three traversal backends.
//!
//! Prints a small markdown table that is pasted verbatim into the
//! research doc and gist. No mocks, no synthetic numbers — every value
//! is measured against a randomly generated Gaussian dataset built at
//! the seed shown in the output.

use std::time::Instant;

use ruvector_hnsw_prefetch::graph::{FlatGraph, GraphConfig};
use ruvector_hnsw_prefetch::search::{brute_force, search, SearchStats, TraversalBackend};

fn main() {
    let cfg = GraphConfig {
        n: 20_000,
        dim: 128,
        m: 16,
        seed: 20260917,
    };
    println!(
        "Building flat HNSW-style graph: n={} dim={} m={} seed={}",
        cfg.n, cfg.dim, cfg.m, cfg.seed
    );
    let t0 = Instant::now();
    let graph = FlatGraph::build(cfg);
    println!("  build: {:.2?}", t0.elapsed());

    // 200 held-out queries drawn from the same distribution.
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(200);
    let mut rng_state: u64 = 0xC0FFEE_1234;
    for _ in 0..200 {
        let mut q = vec![0.0f32; cfg.dim];
        for v in &mut q {
            // Cheap xorshift-based normal.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u1 = ((rng_state & 0xFFFF) as f32 + 1.0) / 65537.0;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u2 = ((rng_state & 0xFFFF) as f32) / 65536.0;
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }
        queries.push(q);
    }

    let k = 10usize;
    let ef = 64usize;

    // Ground truth for recall.
    let t_bf = Instant::now();
    let mut gt: Vec<Vec<u32>> = Vec::with_capacity(queries.len());
    for q in &queries {
        gt.push(brute_force(&graph, q, k));
    }
    println!("  brute-force gt: {:.2?}", t_bf.elapsed());

    println!();
    println!("| Backend        | avg latency (µs) | dist calls / q | recall@10 |");
    println!("|----------------|------------------|----------------|-----------|");

    for backend in [
        TraversalBackend::NoPrefetch,
        TraversalBackend::Prefetch1,
        TraversalBackend::Adaptive,
    ] {
        // Warm-up.
        for q in queries.iter().take(20) {
            let _ = search(&graph, q, k, ef, backend);
        }
        let mut total_ns: u128 = 0;
        let mut total_dist: u64 = 0;
        let mut hits: usize = 0;
        for (q, g) in queries.iter().zip(gt.iter()) {
            let start = Instant::now();
            let (ids, stats) = search(&graph, q, k, ef, backend);
            total_ns += start.elapsed().as_nanos();
            total_dist += stats.distance_calls;
            let gs: std::collections::HashSet<_> = g.iter().collect();
            let is_hit: usize = ids.iter().filter(|i| gs.contains(i)).count();
            hits += is_hit;
        }
        let n = queries.len() as f64;
        let avg_us = (total_ns as f64) / n / 1_000.0;
        let avg_dc = (total_dist as f64) / n;
        let recall = hits as f64 / (n * k as f64);
        println!(
            "| {:<14} | {:>16.2} | {:>14.1} | {:>9.3} |",
            format!("{:?}", backend),
            avg_us,
            avg_dc,
            recall
        );
    }
    println!();
    println!("Numbers above come from `cargo run --release -p ruvector-hnsw-prefetch --bin benchmark`.");
}

#[allow(dead_code)]
fn _unused() -> SearchStats {
    SearchStats {
        distance_calls: 0,
        visited: 0,
        accepted: 0,
        final_prefetch_depth: 0,
    }
}
