//! Benchmark harness for anisotropic PQ.
//!
//! Runs three PQ variants over a synthetic MIPS dataset and reports:
//!   * Recall@10 vs exact MIPS (ground truth via brute-force f32 inner product).
//!   * Score MSE (how close predicted scores are to true inner products).
//!   * Mean per-query latency (µs).
//!   * QPS.
//!   * Codebook memory footprint.
//!   * Mean reconstruction error (L2).
//!
//! All numbers are real, produced by this binary. Reproduce with:
//!     cargo run --release -p ruvector-anisotropic-pq --bin benchmark

use ruvector_anisotropic_pq::{
    anisotropic_pq::AnisotropicPq, dataset::generate, exact_mips, recall_at_k, score_mse,
    standard_pq::StandardPq, Hit, Pq, PqConfig,
};
use std::time::Instant;

struct Row {
    name: String,
    recall_at_10: f32,
    score_mse: f64,
    latency_us_mean: f64,
    latency_us_p95: f64,
    qps: f64,
    memory_kb: f64,
    recon_err: f64,
    train_ms: u128,
}

fn bench_variant(
    mut q: Box<dyn Pq>,
    data: &[f32],
    n: usize,
    queries: &[f32],
    nq: usize,
    dim: usize,
    truth: &[Vec<Hit>],
) -> Row {
    let t0 = Instant::now();
    q.train(data, n);
    let train_ms = t0.elapsed().as_millis();

    let codes = q.encode(data, n);

    let mut latencies = Vec::with_capacity(nq);
    let mut recall_sum = 0.0f64;
    let mut mse_sum = 0.0f64;

    // Warm-up pass (LUT allocation, cache priming).
    for i in 0..nq.min(4) {
        let query = &queries[i * dim..(i + 1) * dim];
        let _ = q.search(query, &codes, n, 10);
    }

    let t_all = Instant::now();
    for i in 0..nq {
        let query = &queries[i * dim..(i + 1) * dim];
        let t = Instant::now();
        let top = q.search(query, &codes, n, 10);
        latencies.push(t.elapsed().as_secs_f64() * 1e6);
        recall_sum += recall_at_k(&top, &truth[i]) as f64;
        mse_sum += score_mse(query, data, &top);
    }
    let elapsed = t_all.elapsed().as_secs_f64();

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let qps = nq as f64 / elapsed;

    let recon = q.reconstruction_error(data, n.min(1024));

    Row {
        name: q.name().to_string(),
        recall_at_10: (recall_sum / nq as f64) as f32,
        score_mse: mse_sum / nq as f64,
        latency_us_mean: mean,
        latency_us_p95: p95,
        qps,
        memory_kb: q.memory_bytes() as f64 / 1024.0,
        recon_err: recon,
        train_ms,
    }
}

fn main() {
    // Dataset knobs.
    let dim = 128usize;
    let n = 10_000usize;
    let nq = 500usize;
    let seed = 0xB4_57_AA_11u64;

    println!("=== ruvector-anisotropic-pq benchmark ===");
    println!("Dataset: n = {}, d = {}, queries = {}, seed = 0x{:x}", n, dim, nq, seed);
    println!("Codebook: m = 8, k = 256, iters = 12");
    println!();
    print!("Generating synthetic MIPS dataset... ");
    let t0 = Instant::now();
    let ds = generate(dim, n, nq, seed);
    println!("done ({} ms)", t0.elapsed().as_millis());

    print!("Computing exact-MIPS ground truth... ");
    let t0 = Instant::now();
    let truth: Vec<Vec<Hit>> = (0..nq)
        .map(|i| exact_mips(&ds.queries[i * dim..(i + 1) * dim], &ds.data, n, 10))
        .collect();
    println!("done ({} ms)", t0.elapsed().as_millis());
    println!();

    // Baseline: exact f32 brute-force. Recall is 1.000 by construction; here
    // we just measure its latency so the PQ variants have a reference point.
    let mut brute_lat = Vec::with_capacity(nq);
    let tb = Instant::now();
    for i in 0..nq {
        let q = &ds.queries[i * dim..(i + 1) * dim];
        let t = Instant::now();
        let _ = exact_mips(q, &ds.data, n, 10);
        brute_lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let brute_elapsed = tb.elapsed().as_secs_f64();
    brute_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let brute_mean = brute_lat.iter().sum::<f64>() / brute_lat.len() as f64;
    let brute_p95 = brute_lat[(brute_lat.len() as f64 * 0.95) as usize];
    let brute_qps = nq as f64 / brute_elapsed;

    let cfg = PqConfig { dim, m: 8, k: 256, iterations: 12, seed };

    let rows: Vec<Row> = vec![
        bench_variant(Box::new(StandardPq::new(cfg)),
            &ds.data, n, &ds.queries, nq, dim, &truth),
        bench_variant(Box::new(AnisotropicPq::new(cfg, 4.0)),
            &ds.data, n, &ds.queries, nq, dim, &truth),
        bench_variant(Box::new(AnisotropicPq::new(cfg, 16.0)),
            &ds.data, n, &ds.queries, nq, dim, &truth),
    ];

    println!("=== Results ===");
    println!();
    println!("{:<28} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "Recall@10", "ScoreMSE", "Mean(µs)", "P95(µs)", "QPS", "Mem(KB)", "Recon(L2)", "Train(ms)");
    println!("{:-<130}", "");
    println!("{:<28} {:>10.3} {:>12.4} {:>10.1} {:>10.1} {:>10.0} {:>10.1} {:>12} {:>10}",
        "ExactBruteForce(f32)",
        1.000f32, 0.0f64, brute_mean, brute_p95, brute_qps,
        (n * dim * 4) as f64 / 1024.0, "-", "-");
    for r in &rows {
        println!("{:<28} {:>10.3} {:>12.4} {:>10.1} {:>10.1} {:>10.0} {:>10.1} {:>12.3} {:>10}",
            r.name, r.recall_at_10, r.score_mse, r.latency_us_mean, r.latency_us_p95,
            r.qps, r.memory_kb, r.recon_err, r.train_ms);
    }
    println!();
    println!("(Recall@10 vs exact f32 MIPS; ScoreMSE = mean squared inner-product error on returned top-10.)");
}
