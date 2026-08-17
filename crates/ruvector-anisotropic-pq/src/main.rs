//! Benchmark: measures recall@10 for inner-product search over synthetic
//! data with mixed norms (Zipf-ish heavy-tail, which mimics real MIPS
//! workloads like recommender item embeddings).
//!
//! Usage:
//!   cargo run --release -p ruvector-anisotropic-pq --bin aniso-pq-bench
//!   ANISO_N=5000 ANISO_DIM=64 ANISO_M=8 ANISO_K=64 cargo run --release ...

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_anisotropic_pq::{
    exact_ip_topk, pq_ip_topk, recall_at_k, AnisotropicTrainer, L2Trainer, NormWeightedTrainer,
    PqCodebook, PqTrainer,
};
use std::time::Instant;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn synth_biased(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    // Base ~ N(0, 1). Multiply by a per-vector heavy-tail scale s_i ~ 1 + Pareto(1.5).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0f32; n * dim];
    for i in 0..n {
        // Approximate Pareto via inverse-CDF from uniform
        let u: f32 = rng.gen_range(1e-4_f32..1.0);
        let scale: f32 = u.powf(-1.0 / 1.5);
        for j in 0..dim {
            // Box-Muller-ish gaussian via two uniforms
            let u1: f32 = rng.gen_range(1e-6_f32..1.0);
            let u2: f32 = rng.gen();
            let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            out[i * dim + j] = g * scale;
        }
    }
    out
}

fn encode_all(cb: &PqCodebook, data: &[f32], n: usize, dim: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * cb.m);
    for i in 0..n {
        out.extend_from_slice(&cb.encode(&data[i * dim..(i + 1) * dim]));
    }
    out
}

fn run_variant(
    name: &str,
    trainer: &dyn PqTrainer,
    base: &[f32],
    queries: &[f32],
    n: usize,
    n_q: usize,
    dim: usize,
    m: usize,
    k: usize,
    topk: usize,
    truths: &[Vec<u32>],
) -> (f64, f64, f64) {
    let t0 = Instant::now();
    let cb = trainer.train(base, n, dim, m, k).expect("train");
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let codes = encode_all(&cb, base, n, dim);
    let enc_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let mut rsum = 0f32;
    for q in 0..n_q {
        let qv = &queries[q * dim..(q + 1) * dim];
        let ret = pq_ip_topk(&cb, &codes, n, qv, topk);
        rsum += recall_at_k(&ret, &truths[q], topk);
    }
    let query_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let recall = (rsum / n_q as f32) as f64;

    println!(
        "{:<18}  recall@{topk} = {:.4}   train={:.1} ms   encode={:.1} ms   query({} q)={:.1} ms   bytes/vec={}",
        name, recall, train_ms, enc_ms, n_q, query_ms, cb.bytes_per_vector()
    );

    let _ = trainer.name();
    (recall, train_ms, query_ms)
}

fn main() {
    let n = env_usize("ANISO_N", 5_000);
    let dim = env_usize("ANISO_DIM", 64);
    let m = env_usize("ANISO_M", 8);
    let k = env_usize("ANISO_K", 64);
    let n_q = env_usize("ANISO_QUERIES", 100);
    let topk = env_usize("ANISO_TOPK", 10);
    let eta_med: f32 = 2.0;
    let eta_hi: f32 = 6.0;

    println!("=== ruvector-anisotropic-pq bench ===");
    println!(
        "n={n} dim={dim} m={m} k={k} d_sub={} n_queries={n_q} topk={topk}",
        dim / m
    );
    println!(
        "compression: {} bytes/vec  vs  {} bytes/vec raw fp32   (ratio {:.1}x)\n",
        m,
        dim * 4,
        (dim * 4) as f32 / m as f32
    );

    let t_data = Instant::now();
    let base = synth_biased(n, dim, 12345);
    let queries = synth_biased(n_q, dim, 67890);
    println!("synth data ready in {:.1} ms", t_data.elapsed().as_secs_f64() * 1000.0);

    // Ground truth
    let t_gt = Instant::now();
    let truths: Vec<Vec<u32>> = (0..n_q)
        .map(|q| exact_ip_topk(&base, n, dim, &queries[q * dim..(q + 1) * dim], topk))
        .collect();
    println!(
        "exact IP ground truth: {:.1} ms  (avg {:.2} ms/query)\n",
        t_gt.elapsed().as_secs_f64() * 1000.0,
        t_gt.elapsed().as_secs_f64() * 1000.0 / n_q as f64
    );

    let r1 = run_variant(
        "L2-PQ",
        &L2Trainer { iters: 20, seed: 1 },
        &base, &queries, n, n_q, dim, m, k, topk, &truths,
    );
    let r2 = run_variant(
        "NormWeighted-PQ",
        &NormWeightedTrainer { iters: 20, seed: 1 },
        &base, &queries, n, n_q, dim, m, k, topk, &truths,
    );
    let r3 = run_variant(
        &format!("Anisotropic η={eta_med}"),
        &AnisotropicTrainer { iters: 20, seed: 1, eta: eta_med },
        &base, &queries, n, n_q, dim, m, k, topk, &truths,
    );
    let r4 = run_variant(
        &format!("Anisotropic η={eta_hi}"),
        &AnisotropicTrainer { iters: 20, seed: 1, eta: eta_hi },
        &base, &queries, n, n_q, dim, m, k, topk, &truths,
    );

    let base_recall = r1.0;
    println!("\n--- deltas vs L2-PQ ---");
    for (label, r) in [
        ("NormWeighted", r2),
        (&format!("Anisotropic η={eta_med}"), r3),
        (&format!("Anisotropic η={eta_hi}"), r4),
    ] {
        let delta = (r.0 - base_recall) * 100.0;
        println!("{:<20} recall Δ = {:+.2} pp    train slowdown = {:.2}x",
            label, delta, r.1 / r1.1);
    }
    println!("\nDone.");
}
