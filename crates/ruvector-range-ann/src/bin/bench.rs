//! Range-ANN benchmark: linear vs IVF-ball-prune vs HNSW-expand-ball.
//!
//! Writes a markdown report to stdout. All numbers are measured, no mocks.

use ruvector_range_ann::data::{calibrate_radius2, make_corpus};
use ruvector_range_ann::hnsw_expand::HnswExpand;
use ruvector_range_ann::ivf::IvfRange;
use ruvector_range_ann::linear::LinearScan;
use ruvector_range_ann::{BackendReport, QueryMeasurement, RangeIndex};
use std::collections::HashSet;
use std::time::Instant;

fn bench_backend<I: RangeIndex>(
    name: &'static str,
    idx: &I,
    queries: &[Vec<f32>],
    r2: f32,
    truth: &[Vec<(u32, f32)>],
    build_time: std::time::Duration,
) -> BackendReport {
    let mut per_query = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let hits = idx.range_search(q, r2);
        let elapsed = t.elapsed();

        let truth_set: HashSet<u32> = truth[qi].iter().map(|(i, _)| *i).collect();
        let got_set: HashSet<u32> = hits.iter().map(|(i, _)| *i).collect();
        let recall = if truth_set.is_empty() {
            1.0
        } else {
            let hit = truth_set.intersection(&got_set).count() as f32;
            hit / truth_set.len() as f32
        };

        per_query.push(QueryMeasurement { elapsed, hits: hits.len(), recall });
    }
    BackendReport { name, build_time, per_query }
}

fn main() {
    // Configuration — modest size so the benchmark finishes in seconds but
    // is still large enough that the three backends have clearly different
    // asymptotics.
    let dim = 48;
    let n_points = 50_000;
    let n_queries = 100;
    let n_clusters = 48;
    let std_dev = 0.06;
    let seed = 2026_11_24;

    println!("# ruvector-range-ann bench\n");
    println!("Configuration:");
    println!("- dim: {dim}");
    println!("- n_points: {n_points}");
    println!("- n_queries: {n_queries}");
    println!("- n_clusters: {n_clusters}");
    println!("- cluster_std: {std_dev}");
    println!("- seed: {seed}\n");

    let t = Instant::now();
    let corpus = make_corpus(seed, n_points, n_queries, dim, n_clusters, std_dev);
    println!("corpus_build_s: {:.3}", t.elapsed().as_secs_f64());

    // Calibrate radius so the ground-truth result set is non-trivial.
    let t = Instant::now();
    let r2 = calibrate_radius2(&corpus.points, &corpus.queries, 60, 8);
    println!("calibrated_r2: {r2:.6}");
    println!("calibration_s: {:.3}\n", t.elapsed().as_secs_f64());

    // --- linear baseline ---
    let t = Instant::now();
    let lin = LinearScan::build(&corpus.points);
    let lin_build = t.elapsed();

    // compute truth via linear
    let mut truth: Vec<Vec<(u32, f32)>> = Vec::with_capacity(corpus.queries.len());
    let t = Instant::now();
    for q in &corpus.queries {
        truth.push(lin.range_search(q, r2));
    }
    let truth_total = t.elapsed();
    println!(
        "ground_truth_total_ms: {:.2}  (per_query_us: {:.1})\n",
        truth_total.as_secs_f64() * 1e3,
        truth_total.as_secs_f64() * 1e6 / corpus.queries.len() as f64
    );

    let lin_report = bench_backend("linear-scan", &lin, &corpus.queries, r2, &truth, lin_build);

    // --- IVF ---
    let t = Instant::now();
    let ivf = IvfRange::build_with(&corpus.points, 160, 8);
    let ivf_build = t.elapsed();
    let ivf_report = bench_backend("ivf-range-ball-prune", &ivf, &corpus.queries, r2, &truth, ivf_build);

    // --- HNSW-expand (two alpha settings so we have three variants beyond baseline) ---
    let t = Instant::now();
    let hnsw_tight = HnswExpand::build_with(&corpus.points, 24, 64, 0.75);
    let hnsw_tight_build = t.elapsed();
    let hnsw_tight_report = bench_backend(
        "hnsw-expand-alpha-0.75",
        &hnsw_tight,
        &corpus.queries,
        r2,
        &truth,
        hnsw_tight_build,
    );

    let t = Instant::now();
    let hnsw_wide = HnswExpand::build_with(&corpus.points, 24, 64, 1.50);
    let hnsw_wide_build = t.elapsed();
    let hnsw_wide_report = bench_backend(
        "hnsw-expand-alpha-1.50",
        &hnsw_wide,
        &corpus.queries,
        r2,
        &truth,
        hnsw_wide_build,
    );

    println!("| backend | build_ms | mean_query_us | mean_hits | mean_recall |");
    println!("|---------|---------:|--------------:|----------:|------------:|");
    for r in [&lin_report, &ivf_report, &hnsw_tight_report, &hnsw_wide_report] {
        println!(
            "| {} | {:.2} | {:.1} | {:.1} | {:.3} |",
            r.name,
            r.build_time.as_secs_f64() * 1e3,
            r.mean_query_us(),
            r.mean_hits(),
            r.mean_recall()
        );
    }
    println!();

    // Speedups
    let base = lin_report.mean_query_us();
    println!("Speedup over linear scan (mean_query_us):");
    for r in [&ivf_report, &hnsw_tight_report, &hnsw_wide_report] {
        let sp = base / r.mean_query_us();
        println!("- {}: {:.2}x", r.name, sp);
    }
}
