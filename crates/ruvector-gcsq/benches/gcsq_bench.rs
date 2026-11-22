//! Standalone benchmark binary (no criterion dep).
//!
//! Measures encode throughput and reranker query throughput for GCSQ, Binary SQ,
//! and a bit-matched SimHash, plus Spearman/recall quality. Prints a single
//! Markdown table ready for the research doc.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_gcsq::backends::{evaluate, BinarySqReranker, GcsqReranker, Reranker, SimHashReranker};
use std::time::Instant;

fn make(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let u1: f32 = rng.gen_range(1e-9f32..1.0);
                    let u2: f32 = rng.gen_range(0.0f32..1.0);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect()
        })
        .collect()
}

struct Row {
    name: &'static str,
    bits_per_dim: u8,
    code_bytes: usize,
    encode_ns_per_vec: u128,
    rerank_ns_per_pair: u128,
    spearman: f64,
    recall_at_10: f64,
    bytes_per_vec_total: usize,
}

fn bench_backend(
    name: &'static str,
    bits_per_dim: u8,
    backend: &dyn Reranker,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> Row {
    // Encode throughput.
    let start = Instant::now();
    let base_codes: Vec<Vec<u8>> = base.iter().map(|v| backend.encode(v)).collect();
    let encode_elapsed = start.elapsed().as_nanos();
    let encode_ns_per_vec = encode_elapsed / base.len() as u128;

    // Rerank throughput: each query scans all base codes.
    let query_codes: Vec<Vec<u8>> = queries.iter().map(|v| backend.encode(v)).collect();
    let start = Instant::now();
    let mut sink = 0u64;
    for qc in &query_codes {
        for bc in &base_codes {
            sink = sink.wrapping_add(backend.distance(qc, bc) as u64);
        }
    }
    std::hint::black_box(sink);
    let rerank_elapsed = start.elapsed().as_nanos();
    let total_pairs = (queries.len() * base.len()) as u128;
    let rerank_ns_per_pair = rerank_elapsed / total_pairs;

    let q = evaluate(backend, base, queries, 10);
    Row {
        name,
        bits_per_dim,
        code_bytes: backend.code_bytes(),
        encode_ns_per_vec,
        rerank_ns_per_pair,
        spearman: q.spearman_rho,
        recall_at_10: q.recall_at_k,
        bytes_per_vec_total: backend.code_bytes(),
    }
}

fn run_suite(n: usize, dim: usize, queries_n: usize) {
    println!(
        "\n## GCSQ bench · n={} base × {} queries, dim={}\n",
        n, queries_n, dim
    );
    let base = make(n, dim, 1);
    let queries = make(queries_n, dim, 2);

    let mut rows: Vec<Row> = Vec::new();
    for bits in [1u8, 2, 4] {
        let gc = GcsqReranker::fit(&base, bits);
        let bn = BinarySqReranker::fit(&base, bits);
        rows.push(bench_backend("gcsq", bits, &gc, &base, &queries));
        rows.push(bench_backend("binary_sq", bits, &bn, &base, &queries));
    }
    for n_bits_mult in [1usize, 2, 4] {
        let n_bits = dim * n_bits_mult;
        let sh = SimHashReranker::fit(dim, n_bits, 99);
        rows.push(bench_backend(
            "simhash",
            n_bits_mult as u8,
            &sh,
            &base,
            &queries,
        ));
    }

    println!(
        "| backend    | bits/dim | code_bytes | encode ns/vec | rerank ns/pair | spearman ρ | recall@10 |"
    );
    println!(
        "|------------|----------|------------|---------------|----------------|------------|-----------|"
    );
    for r in rows {
        println!(
            "| {:<10} | {:>8} | {:>10} | {:>13} | {:>14} | {:>10.4} | {:>9.4} |",
            r.name,
            r.bits_per_dim,
            r.code_bytes,
            r.encode_ns_per_vec,
            r.rerank_ns_per_pair,
            r.spearman,
            r.recall_at_10
        );
        let _ = r.bytes_per_vec_total; // included for completeness
    }
}

fn main() {
    println!("# ruvector-gcsq benchmark (release build)");
    run_suite(5000, 64, 100);
    run_suite(5000, 128, 100);
    run_suite(10000, 128, 100);
}
