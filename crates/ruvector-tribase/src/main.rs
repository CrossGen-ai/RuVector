//! Tribase demo / benchmark binary.
//!
//! Runs three indices (flat, plain IVF, Tribase IVF) on a synthetic
//! Gaussian-cluster dataset, measures recall@10 against the flat
//! baseline, query latency, and pruning ratio.
//!
//! Run with `cargo run --release -p ruvector-tribase --bin tribase-demo`.

use ruvector_tribase::{
    make_clustered, AnnIndex, FlatIndex, PlainIvfIndex, SearchStats, TribaseIndex,
};
use std::time::Instant;

fn pct(n: u64, d: u64) -> f32 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f32 / d as f32
    }
}

fn recall_at(truth: &[u32], got: &[u32]) -> f32 {
    let k = truth.len() as f32;
    let mut hit = 0u32;
    for g in got {
        if truth.contains(g) {
            hit += 1;
        }
    }
    hit as f32 / k
}

fn bench<I: AnnIndex + ?Sized>(
    name: &str,
    idx: &I,
    queries: &[Vec<f32>],
    k: usize,
    truth: &[Vec<u32>],
) {
    let mut total_recall = 0.0f32;
    let mut stats = SearchStats::default();
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let (r, s) = idx.search_with_stats(q, k);
        let ids: Vec<u32> = r.iter().map(|n| n.id).collect();
        total_recall += recall_at(&truth[qi], &ids);
        stats.merge(&s);
    }
    let elapsed = t0.elapsed();
    let qps = queries.len() as f32 / elapsed.as_secs_f32();
    let per_q_us = elapsed.as_micros() as f32 / queries.len() as f32;
    println!(
        "  {:<14}  recall@{}={:.3}  qps={:>8.0}  us/q={:>7.1}  full_dist/q={:>6.0}  prune%={:.1}  mem={}KiB",
        name,
        k,
        total_recall / queries.len() as f32,
        qps,
        per_q_us,
        stats.full_dist as f32 / queries.len() as f32,
        pct(stats.pruned, stats.considered),
        idx.estimated_bytes() / 1024,
    );
}

fn main() {
    let n: usize = 50_000;
    let dim: usize = 64;
    let n_centers = 80;
    let spread = 0.04;
    let n_queries = 500;
    let k = 10;

    println!("ruvector-tribase demo");
    println!(
        "n={n} dim={dim} n_centers={n_centers} spread={spread} queries={n_queries} k={k}"
    );

    let data = make_clustered(n, dim, n_centers, spread, 42);
    let queries: Vec<Vec<f32>> = make_clustered(n_queries, dim, n_centers, spread, 99);

    // Ground truth from flat.
    let t0 = Instant::now();
    let flat = FlatIndex::new(data.clone());
    println!("Built flat in {:?}", t0.elapsed());

    let truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| flat.search(q, k).into_iter().map(|n| n.id).collect())
        .collect();

    // Plain IVF and Tribase share the same probe count for a fair comparison.
    for &n_probe in &[4usize, 8, 16] {
        println!("--- n_probe = {n_probe} ---");

        bench("flat", &flat, &queries, k, &truth);

        let t0 = Instant::now();
        let plain = PlainIvfIndex::build(data.clone(), 128, n_probe, 18, 11);
        let t_plain = t0.elapsed();

        let t0 = Instant::now();
        let tri = TribaseIndex::build(data.clone(), 128, n_probe, 18, 11);
        let t_tri = t0.elapsed();

        println!(
            "  built plain in {:?}, tribase in {:?} (128 clusters)",
            t_plain, t_tri
        );

        bench("ivf-plain", &plain, &queries, k, &truth);
        bench("ivf-tribase", &tri, &queries, k, &truth);
    }
}
