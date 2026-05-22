//! symphonyqg-demo: produce the real numbers cited in the research doc.
//!
//! Builds three indexes on a synthetic dataset and prints recall + per-query
//! latency for each:
//!   1. brute force (reference for recall)
//!   2. NSW graph with full-precision distance traversal
//!   3. SymphonyQG: NSW graph + quantized traversal + full re-rank

use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_symphonyqg::{SymphonyQg, SymphonyQgParams};
use ruvector_symphonyqg::index::brute_force;

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect()).collect()
}

fn measure_recall(got: &[(u32, f32)], gt: &std::collections::HashSet<u32>) -> usize {
    got.iter().filter(|(i, _)| gt.contains(i)).count()
}

fn run(n: usize, d: usize, m: usize, ef: usize, n_queries: usize, k: usize) {
    println!("\n=== n={} d={} M={} ef={} k={} ===", n, d, m, ef, k);
    let db = synth(n, d, 42);
    let queries = synth(n_queries, d, 99);

    let t0 = Instant::now();
    let idx = SymphonyQg::build(db.clone(), SymphonyQgParams {
        m, ef_construction: ef, ef_search: ef, rotation_seed: 0xA5A5_5A5A,
    });
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (vec_bytes, code_bytes) = idx.memory_bytes();
    println!("  build:                 {:.1} ms", build_ms);
    println!("  full-precision bytes:  {} ({:.2} MB)", vec_bytes, vec_bytes as f64 / 1e6);
    println!("  bit-code bytes:        {} ({:.2} MB, {:.1}x smaller)",
        code_bytes, code_bytes as f64 / 1e6, vec_bytes as f64 / code_bytes as f64);

    // Ground truth
    let gts: Vec<std::collections::HashSet<u32>> = queries.iter()
        .map(|q| brute_force(&db, q, k).into_iter().map(|(i,_)| i).collect())
        .collect();

    // 1) Brute force latency
    let t = Instant::now();
    for q in &queries { let _ = brute_force(&db, q, k); }
    let brute_us = t.elapsed().as_micros() as f64 / queries.len() as f64;

    // 2) NSW with exact distance
    let mut hits = 0usize;
    let t = Instant::now();
    for (q, gt) in queries.iter().zip(&gts) {
        let r = idx.search_exact_graph(q, k);
        hits += measure_recall(&r, gt);
    }
    let exact_graph_us = t.elapsed().as_micros() as f64 / queries.len() as f64;
    let exact_graph_recall = hits as f32 / (queries.len() * k) as f32;

    // 3) SymphonyQG
    let mut hits = 0usize;
    let t = Instant::now();
    for (q, gt) in queries.iter().zip(&gts) {
        let r = idx.search(q, k);
        hits += measure_recall(&r, gt);
    }
    let sqg_us = t.elapsed().as_micros() as f64 / queries.len() as f64;
    let sqg_recall = hits as f32 / (queries.len() * k) as f32;

    println!("  brute-force latency:        {:.1} us/query", brute_us);
    println!("  NSW exact-graph latency:    {:.1} us/query   recall@{}={:.3}", exact_graph_us, k, exact_graph_recall);
    println!("  SymphonyQG (FP estim) lat:  {:.1} us/query   recall@{}={:.3}", sqg_us, k, sqg_recall);

    // 4) SymphonyQG-Fast (popcount)
    let mut hits = 0usize;
    let t = Instant::now();
    for (q, gt) in queries.iter().zip(&gts) {
        let r = idx.search_popcount(q, k);
        hits += measure_recall(&r, gt);
    }
    let pc_us = t.elapsed().as_micros() as f64 / queries.len() as f64;
    let pc_recall = hits as f32 / (queries.len() * k) as f32;
    println!("  SymphonyQG (popcount) lat:  {:.1} us/query   recall@{}={:.3}", pc_us, k, pc_recall);

    println!("  popcount speedup:           {:.1}x vs brute, {:.1}x vs exact-graph",
        brute_us / pc_us, exact_graph_us / pc_us);
}

fn main() {
    println!("ruvector-symphonyqg PoC benchmark");
    println!("(release build recommended: cargo run --release -p ruvector-symphonyqg)");
    run(2_000,  64,  16, 64, 200, 10);
    run(5_000, 128,  16, 64, 200, 10);
    run(10_000, 128, 24, 96, 200, 10);
}
