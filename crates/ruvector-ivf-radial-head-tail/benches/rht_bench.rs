//! Real cargo-run benchmark comparing:
//!   (A) Exact brute-force fp32 (baseline for recall)
//!   (B) IVF with uniform SQ8 (baseline backend)
//!   (C) IVF-RHT Full
//!   (D) IVF-RHT Adaptive
//!
//! Measures: memory bytes, build time, QPS, mean recall@10.
//!
//! Run with:  cargo run --release -p ruvector-ivf-radial-head-tail --bench rht_bench

use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use ruvector_ivf_radial_head_tail::{baseline::IvfSq8, exact_topk, recall_at_k, BuildParams, IvfRht, ScanMode};
use std::time::Instant;

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    // Mixture-of-gaussians to give real cluster structure.
    let n_clusters = 32usize;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..n_clusters)];
            (0..dim)
                .map(|d| c[d] + (rng.gen::<f32>() - 0.5) * 0.4)
                .collect()
        })
        .collect()
}

fn main() {
    let n = 20_000usize;
    let dim = 64usize;
    let n_queries = 500usize;
    let k = 10usize;
    let nprobe = 8usize;

    println!("=== IVF-RHT benchmark ===");
    println!("dataset: {n} vectors x {dim} dims (mixture-of-32-gaussians)");
    println!("queries: {n_queries}, k={k}, nprobe={nprobe}\n");

    let data = gen_data(n, dim, 42);
    let queries = gen_data(n_queries, dim, 1337);

    println!("Computing ground truth (exact brute-force)...");
    let t0 = Instant::now();
    let truths: Vec<Vec<(u32, f32)>> = queries.iter().map(|q| exact_topk(&data, q, k)).collect();
    let gt_ms = t0.elapsed().as_millis();
    println!("  ground-truth time: {gt_ms} ms ({:.2} ms/query)\n", gt_ms as f32 / n_queries as f32);

    let params = BuildParams {
        n_cells: 128,
        head_fraction: 0.5,
        kmeans_iters: 10,
        seed: 2026,
    };

    // ---- Baseline IVF-SQ8 ----
    let t0 = Instant::now();
    let base = IvfSq8::build(&data, params);
    let build_sq8_ms = t0.elapsed().as_millis();
    let mem_sq8 = base.memory_bytes();
    let t0 = Instant::now();
    let mut recall_sq8 = 0.0f32;
    for (q, gt) in queries.iter().zip(&truths) {
        let r = base.search(q, k, nprobe);
        recall_sq8 += recall_at_k(&r, gt);
    }
    let qms_sq8 = t0.elapsed().as_micros() as f32 / 1000.0;
    recall_sq8 /= n_queries as f32;

    // ---- IVF-RHT (head 50%) ----
    let t0 = Instant::now();
    let rht = IvfRht::build(&data, params);
    let build_rht_ms = t0.elapsed().as_millis();
    let mem_rht = rht.memory_bytes();
    let t0 = Instant::now();
    let mut recall_full = 0.0f32;
    for (q, gt) in queries.iter().zip(&truths) {
        let r = rht.search(q, k, nprobe, ScanMode::Full);
        recall_full += recall_at_k(&r, gt);
    }
    let qms_full = t0.elapsed().as_micros() as f32 / 1000.0;
    recall_full /= n_queries as f32;

    let t0 = Instant::now();
    let mut recall_adaptive = 0.0f32;
    for (q, gt) in queries.iter().zip(&truths) {
        let r = rht.search(q, k, nprobe, ScanMode::Adaptive);
        recall_adaptive += recall_at_k(&r, gt);
    }
    let qms_adaptive = t0.elapsed().as_micros() as f32 / 1000.0;
    recall_adaptive /= n_queries as f32;

    let t0 = Instant::now();
    let mut recall_head = 0.0f32;
    for (q, gt) in queries.iter().zip(&truths) {
        let r = rht.search(q, k, nprobe, ScanMode::HeadOnly);
        recall_head += recall_at_k(&r, gt);
    }
    let qms_head = t0.elapsed().as_micros() as f32 / 1000.0;
    recall_head /= n_queries as f32;

    let qps = |ms: f32| (n_queries as f32) / (ms / 1000.0);

    println!("{:-<78}", "");
    println!(
        "{:<28}  {:>10}  {:>10}  {:>10}  {:>8}",
        "variant", "mem MB", "build ms", "qps", "recall@10"
    );
    println!("{:-<78}", "");
    println!(
        "{:<28}  {:>10.3}  {:>10}  {:>10.0}  {:>8.4}",
        "IVF uniform SQ8",
        mem_sq8 as f32 / 1e6,
        build_sq8_ms,
        qps(qms_sq8),
        recall_sq8
    );
    println!(
        "{:<28}  {:>10.3}  {:>10}  {:>10.0}  {:>8.4}",
        "IVF-RHT Full (ours)",
        mem_rht as f32 / 1e6,
        build_rht_ms,
        qps(qms_full),
        recall_full
    );
    println!(
        "{:<28}  {:>10.3}  {:>10}  {:>10.0}  {:>8.4}",
        "IVF-RHT Adaptive (ours)",
        mem_rht as f32 / 1e6,
        build_rht_ms,
        qps(qms_adaptive),
        recall_adaptive
    );
    println!(
        "{:<28}  {:>10.3}  {:>10}  {:>10.0}  {:>8.4}",
        "IVF-RHT HeadOnly (ours)",
        mem_rht as f32 / 1e6,
        build_rht_ms,
        qps(qms_head),
        recall_head
    );
    println!("{:-<78}", "");

    println!(
        "\nMemory savings vs SQ8: {:.1}%",
        100.0 * (1.0 - mem_rht as f32 / mem_sq8 as f32)
    );
    println!("Full recall gap vs SQ8: {:+.4}", recall_full - recall_sq8);
    println!("Adaptive speedup vs Full: {:.2}x", qms_full / qms_adaptive.max(1e-3));
}
