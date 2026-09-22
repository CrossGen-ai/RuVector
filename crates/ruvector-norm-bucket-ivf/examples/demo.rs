//! Runnable demo: builds an IVF index, queries with three backends, and
//! prints recall, timings, and prune stats.
//!
//! `cargo run -p ruvector-norm-bucket-ivf --release --example demo`

use ruvector_norm_bucket_ivf::{
    brute_force_topk, gaussian_dataset, mixed_scale_dataset, recall_at_k, FullScan, IvfIndex,
    NormBucketPartialL2Scan, NormBucketScan, ScanBackend, ScanStats,
};
use std::time::Instant;

fn run<B: ScanBackend>(
    label: &str,
    idx: &IvfIndex,
    queries: &[f32],
    truth: &[Vec<(u32, f32)>],
    k: usize,
    nprobe: usize,
    dim: usize,
    backend: &B,
) {
    let mut agg = ScanStats::default();
    let mut recall_sum = 0f32;
    let start = Instant::now();
    for (qi, chunk) in queries.chunks_exact(dim).enumerate() {
        let (got, s) = idx.search_with_stats(chunk, k, nprobe, backend);
        recall_sum += recall_at_k(&got, &truth[qi]);
        agg.candidates += s.candidates;
        agg.full_distances += s.full_distances;
        agg.skipped_by_norm += s.skipped_by_norm;
        agg.early_abandoned += s.early_abandoned;
    }
    let elapsed = start.elapsed();
    let nq = queries.len() / dim;
    let recall = recall_sum / nq as f32;
    let qps = nq as f64 / elapsed.as_secs_f64();
    println!(
        "  {:<28} recall@{k}={:.3}  qps={:>9.1}  full_L2={:>9}  skipped_by_norm={:>9}  early_abandoned={:>9}",
        label,
        recall,
        qps,
        agg.full_distances,
        agg.skipped_by_norm,
        agg.early_abandoned
    );
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let nq: usize = std::env::var("NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let clusters: usize = std::env::var("CLUSTERS").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let nprobe: usize = std::env::var("NPROBE").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("ruvector-norm-bucket-ivf demo");
    println!("  n={n}  dim={dim}  nq={nq}  k={k}  clusters={clusters}  nprobe={nprobe}");

    let dist = std::env::var("DIST").unwrap_or_else(|_| "mixed".to_string());
    let (vectors, queries) = if dist == "gauss" {
        (
            gaussian_dataset(n, dim, 42),
            gaussian_dataset(nq, dim, 1337),
        )
    } else {
        (
            mixed_scale_dataset(n, dim, 42, 0.2, 5.0),
            mixed_scale_dataset(nq, dim, 1337, 0.2, 5.0),
        )
    };
    println!("  dataset: {dist}");

    let t = Instant::now();
    let idx = IvfIndex::build(&vectors, dim, clusters, 15, 7);
    println!(
        "  build: {:.2}s  index_bytes~={} MiB",
        t.elapsed().as_secs_f64(),
        idx.estimated_bytes() / (1024 * 1024)
    );

    // Ground truth
    let mut truth = Vec::with_capacity(nq);
    for chunk in queries.chunks_exact(dim) {
        truth.push(brute_force_topk(&vectors, dim, chunk, k));
    }

    run("baseline full_scan", &idx, &queries, &truth, k, nprobe, dim, &FullScan);
    run("norm_bucket", &idx, &queries, &truth, k, nprobe, dim, &NormBucketScan);
    run("norm_bucket+partial_l2", &idx, &queries, &truth, k, nprobe, dim, &NormBucketPartialL2Scan);
}
