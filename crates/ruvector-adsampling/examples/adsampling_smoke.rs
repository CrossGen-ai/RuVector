//! End-to-end smoke: build a rotated brute-force index over synthetic
//! data, run each oracle, print the report. Used to capture the
//! numbers into the research README.

use rand::{Rng, SeedableRng};
use ruvector_adsampling::bench_variants;

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| r.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

fn main() {
    let corpus_n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_192);
    let dim: usize = std::env::var("D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let queries_n: usize = std::env::var("Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!(
        "# adsampling-smoke  N={corpus_n} D={dim} Q={queries_n} K={k}"
    );
    let corpus = synth(corpus_n, dim, 0xABCD);
    let queries = synth(queries_n, dim, 0x1234);
    let rows = bench_variants(&corpus, &queries, k, 42);
    println!(
        "| variant       |  qps   | ops/query | recall@k | prune rate |"
    );
    println!(
        "|---------------|--------|-----------|----------|------------|"
    );
    for r in &rows {
        println!(
            "| {:<13} | {:>6.1} | {:>9.0} | {:>8.3} | {:>10.3} |",
            r.name, r.qps, r.ops_per_query, r.recall_at_k, r.prune_rate
        );
    }
}
