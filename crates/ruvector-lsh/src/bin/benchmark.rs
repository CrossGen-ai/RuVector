//! Real benchmark harness — three variants vs brute-force ground truth.
//!
//! Run:  cargo run --release -p ruvector-lsh --bin lsh_benchmark
//!
//! Outputs a Markdown-friendly table to stdout with measured numbers:
//!   build_ms, query_us (mean), recall@10, candidates (mean).

use ruvector_lsh::*;
use std::time::Instant;

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    v[i]
}

fn bench<I: AnnIndex>(idx: &I, queries: &[Vec<f32>], truth: &[Vec<(usize, f32)>], k: usize) -> (f64, f64, f64) {
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut recalls = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let res = idx.search(q, k);
        latencies_us.push(t.elapsed().as_secs_f64() * 1_000_000.0);
        recalls.push(recall_at_k(&res, &truth[qi], k) as f64);
    }
    let mean_us = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let p95_us = percentile(latencies_us, 0.95);
    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    (mean_us, p95_us, mean_recall)
}

fn main() {
    let n = std::env::var("LSH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000usize);
    let dim = std::env::var("LSH_D").ok().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    let nq = std::env::var("LSH_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200usize);
    let k = 10;

    println!("# ruvector-lsh benchmark");
    println!("dataset: n={n}, dim={dim}, queries={nq}, k={k}");
    println!();

    println!("generating clustered data (embedding-like)...");
    let data = gen_clustered_dataset(n, dim, 200, 0.15, 42);
    let queries = gen_queries_near(&data, nq, 0.10, 7);

    // Ground truth via brute force (also used as variant #1).
    println!("computing ground truth (brute-force)...");
    let bf = BruteForce::new(data.clone());
    let t = Instant::now();
    let truth: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| bf.search(q, k)).collect();
    let bf_total_us = t.elapsed().as_secs_f64() * 1_000_000.0;
    let bf_mean_us = bf_total_us / nq as f64;

    println!();
    println!("| variant | build_ms | mean_us | p95_us | recall@10 |");
    println!("|---|---:|---:|---:|---:|");
    println!("| BruteForce | 0 | {:.1} | - | 1.000 |", bf_mean_us);

    // Variant 2: SimHash LSH at multiple bit widths.
    for bits in [32usize, 64, 128] {
        let t = Instant::now();
        let idx = SimHashLsh::build(data.clone(), bits, 123);
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (mean_us, p95_us, r) = bench(&idx, &queries, &truth, k);
        println!(
            "| SimHashLsh(b={bits}) | {:.1} | {:.1} | {:.1} | {:.3} |",
            build_ms, mean_us, p95_us, r
        );
    }

    // Variant 3: Multi-probe SimHash with varying L and probes.
    for (bits, l, probes) in [(32usize, 4usize, 4usize), (32, 8, 8), (64, 4, 8)] {
        let t = Instant::now();
        let idx = MultiProbeSimHash::build(data.clone(), bits, l, probes, 321);
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (mean_us, p95_us, r) = bench(&idx, &queries, &truth, k);
        println!(
            "| MultiProbeSimHash(b={bits},L={l},probes={probes}) | {:.1} | {:.1} | {:.1} | {:.3} |",
            build_ms, mean_us, p95_us, r
        );
    }

    println!();
    println!("notes: latencies are wallclock for single-threaded queries on the current host.");
}
