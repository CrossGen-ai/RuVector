//! Runnable end-to-end benchmark for LVQ variants.
//!
//! Generates a synthetic corpus of Gaussian-cluster vectors (dim=128,
//! N=10_000, C=32 clusters), encodes with fp32 / LVQ8 / LVQ4 / LVQ4x8,
//! then measures:
//!
//! - Per-vector code bytes (real, from `bytes_per_code`).
//! - Encode time per vector.
//! - Asymmetric distance latency per (query, base) pair.
//! - Recall@10 (approx top-1 in exact top-10) over 100 random queries.
//!
//! Prints a plain-text table. Deterministic given seed. No mocks; every
//! reported number comes from real measurement.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_lvq::{l2_sq_f32, recall_at_k, Lvq4, Lvq4x8, Lvq8, Quantizer};
use std::time::Instant;

fn gen_corpus(n: usize, d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Random cluster centers uniform in [-1, 1]^d.
    let centers: Vec<Vec<f32>> = (0..c)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let noise = Normal::new(0.0f32, 0.1).unwrap();
    (0..n)
        .map(|i| {
            let ctr = &centers[i % c];
            (0..d).map(|j| ctr[j] + noise.sample(&mut rng)).collect()
        })
        .collect()
}

fn time_encode<Q: Quantizer>(q: &Q, corpus: &[Vec<f32>]) -> (Vec<Q::Code>, f64) {
    let t0 = Instant::now();
    let codes: Vec<Q::Code> = corpus.iter().map(|v| q.encode(v)).collect();
    let elapsed_ns = t0.elapsed().as_nanos() as f64;
    (codes, elapsed_ns / corpus.len() as f64)
}

fn time_asym<Q: Quantizer>(q: &Q, queries: &[Vec<f32>], codes: &[Q::Code]) -> f64 {
    // Warm-up.
    let mut sink = 0.0f32;
    for qv in queries.iter().take(4) {
        for c in codes.iter().take(64) {
            sink += q.asymmetric_l2_sq(qv, c);
        }
    }
    let t0 = Instant::now();
    for qv in queries {
        for c in codes {
            sink += q.asymmetric_l2_sq(qv, c);
        }
    }
    let elapsed_ns = t0.elapsed().as_nanos() as f64;
    // Prevent optimizer eliding the loop.
    if sink == f32::INFINITY {
        eprintln!("sink=inf");
    }
    elapsed_ns / (queries.len() * codes.len()) as f64
}

fn recall_report<Q: Quantizer>(
    label: &str,
    q: &Q,
    corpus: &[Vec<f32>],
    codes: &[Q::Code],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let recall = recall_at_k(
        queries.len(),
        corpus.len(),
        &|qi, bi| l2_sq_f32(&queries[qi], &corpus[bi]),
        |qi, bi| q.asymmetric_l2_sq(&queries[qi], &codes[bi]),
        k,
    );
    println!("  Recall@{k} {label}: {:.4}", recall);
    recall
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let qn: usize = std::env::var("Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let k: usize = 10;
    let seed: u64 = 42;

    println!("=== ruvector-lvq bench ===");
    println!("Corpus: N={n}, dim={d}, queries={qn}, seed={seed}");
    let corpus = gen_corpus(n, d, 32, seed);
    let queries = gen_corpus(qn, d, 32, seed.wrapping_add(1));

    // fp32 baseline
    let fp32_bytes = d * 4;
    println!("\n-- fp32 baseline --");
    println!("  bytes/vec: {fp32_bytes}");
    // fp32 asym latency (using exact L2 as the "quantizer" — no encode).
    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for qv in &queries {
        for c in &corpus {
            sink += l2_sq_f32(qv, c);
        }
    }
    let fp32_asym_ns = t0.elapsed().as_nanos() as f64 / (queries.len() * corpus.len()) as f64;
    if sink == f32::INFINITY {
        eprintln!("sink=inf");
    }
    println!("  L2 latency: {fp32_asym_ns:.1} ns / pair");

    // LVQ8
    println!("\n-- LVQ-8 --");
    let q8 = Lvq8::new(d);
    let (c8, e8_ns) = time_encode(&q8, &corpus);
    println!(
        "  bytes/vec: {} (ratio {:.3}x fp32)",
        q8.bytes_per_code(),
        q8.bytes_per_code() as f32 / fp32_bytes as f32
    );
    println!("  encode: {e8_ns:.1} ns/vec");
    let a8_ns = time_asym(&q8, &queries, &c8);
    println!("  asym L2 latency: {a8_ns:.1} ns / pair");
    recall_report("LVQ8", &q8, &corpus, &c8, &queries, k);

    // LVQ4
    println!("\n-- LVQ-4 --");
    let q4 = Lvq4::new(d);
    let (c4, e4_ns) = time_encode(&q4, &corpus);
    println!(
        "  bytes/vec: {} (ratio {:.3}x fp32)",
        q4.bytes_per_code(),
        q4.bytes_per_code() as f32 / fp32_bytes as f32
    );
    println!("  encode: {e4_ns:.1} ns/vec");
    let a4_ns = time_asym(&q4, &queries, &c4);
    println!("  asym L2 latency: {a4_ns:.1} ns / pair");
    recall_report("LVQ4", &q4, &corpus, &c4, &queries, k);

    // LVQ4x8
    println!("\n-- LVQ-4x8 (residual) --");
    let q48 = Lvq4x8::new(d);
    let (c48, e48_ns) = time_encode(&q48, &corpus);
    println!(
        "  bytes/vec: {} (ratio {:.3}x fp32)",
        q48.bytes_per_code(),
        q48.bytes_per_code() as f32 / fp32_bytes as f32
    );
    println!("  encode: {e48_ns:.1} ns/vec");
    let a48_ns = time_asym(&q48, &queries, &c48);
    println!("  asym L2 latency: {a48_ns:.1} ns / pair");
    recall_report("LVQ4x8", &q48, &corpus, &c48, &queries, k);

    println!("\n=== summary ===");
    println!(
        "  ratio: fp32=1.000x, lvq8={:.3}x, lvq4={:.3}x, lvq4x8={:.3}x",
        q8.bytes_per_code() as f32 / fp32_bytes as f32,
        q4.bytes_per_code() as f32 / fp32_bytes as f32,
        q48.bytes_per_code() as f32 / fp32_bytes as f32,
    );
}
