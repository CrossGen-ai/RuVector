//! Anytime HNSW — benchmark binary.
//!
//! Measures three search variants on a clustered flat k-NN proximity graph and
//! reports:
//!
//! * Final recall@k (must match across variants — anytime preserves the answer)
//! * Per-query latency p50/p95/p99 (impact of snapshot overhead)
//! * Snapshot count per query (reactivity)
//! * **Time-to-quality** curve: median wall-clock time to reach
//!   `0.50`, `0.70`, `0.90`, `0.95`, `0.99 × final-recall` thresholds.
//!
//! ## Acceptance criteria
//!
//! 1. Final recall@k matches one-shot baseline (anytime must not perturb).
//! 2. Top-k monotonicity holds for every snapshot across every query.
//! 3. Batched anytime emits strictly fewer snapshots than naive while
//!    reaching the same final result.
//!
//! ## Usage
//!
//!   cargo run --release -p ruvector-anytime --bin benchmark

use std::time::{Duration, Instant};

use ruvector_anytime::{
    dataset::{clustered_queries, clustered_unit_vectors, ground_truth},
    graph::{FlatGraph, GraphConfig},
    metrics::{memory_estimate_bytes, recall_at_k, LatencyStats},
    search::{
        AnytimeSnapshot, BatchedAnytime, NaiveAnytime, OneShotSearch, SearchResult,
        Searcher,
    },
};

// ─── Dataset parameters ───────────────────────────────────────────────────────
const N_CLUSTERS: usize = 10;
const N_PER_CLUSTER: usize = 300; // 3000 total
const N: usize = N_CLUSTERS * N_PER_CLUSTER;
const DIMS: usize = 48;
const CLUSTER_STD: f32 = 0.12;

const M: usize = 16;
const M_LONGJUMP: usize = 6;
const K: usize = 10;
const EF: usize = 120;
const N_QUERIES: usize = 200;
const ENTRY: usize = 0;

// Recall-fraction thresholds for the time-to-quality curve.
const TTQ_FRACTIONS: &[f32] = &[0.50, 0.70, 0.90, 0.95, 0.99];

// ─── Acceptance thresholds ────────────────────────────────────────────────────
const MIN_BASELINE_RECALL: f32 = 0.80;
// Anytime variants must match the baseline final recall to within 1e-6.
const RECALL_EXACT_TOL: f32 = 1e-6;
// Batched should emit at least 4× fewer snapshots than naive in aggregate.
const MIN_BATCHED_REDUCTION: f32 = 4.0;

fn main() {
    print_header();

    eprintln!("[bench] Generating clustered dataset: {N_CLUSTERS} × {N_PER_CLUSTER} = {N} vectors, D={DIMS}");
    let (data, _assign) =
        clustered_unit_vectors(N_CLUSTERS, N_PER_CLUSTER, DIMS, CLUSTER_STD, 0xDEAD_BEEF);

    eprintln!("[bench] Generating {N_QUERIES} clustered queries");
    let queries = clustered_queries(
        N_QUERIES, DIMS, &data, N_PER_CLUSTER, CLUSTER_STD, 0xCAFE_BABE,
    );

    eprintln!("[bench] Computing brute-force ground truth (K={K})");
    let gt = ground_truth(&data, &queries, DIMS, K);

    eprintln!("[bench] Building flat graph (M={M}, M_LJ={M_LONGJUMP})");
    let t0 = Instant::now();
    let graph = FlatGraph::build(
        data.clone(),
        GraphConfig {
            m: M,
            m_longjump: M_LONGJUMP,
            dims: DIMS,
        },
    );
    let build_ms = t0.elapsed().as_millis();
    eprintln!("[bench] Graph built in {build_ms} ms");

    let mem = memory_estimate_bytes(N, DIMS, M + M_LONGJUMP);
    println!(
        "Memory estimate: {:.2} MiB (vectors + adjacency, N={N}, D={DIMS}, deg={})",
        mem as f64 / (1024.0 * 1024.0),
        M + M_LONGJUMP
    );
    println!("Build time: {build_ms} ms");
    println!("Entry: node {ENTRY} (fixed — simulates HNSW layer-0 cold start)");
    println!();

    // ─── Run benchmarks ───────────────────────────────────────────────────────
    let oneshot = run_variant("OneShot", &graph, &queries, &gt, &OneShotSearch, false);
    let naive = run_variant("NaiveAnytime", &graph, &queries, &gt, &NaiveAnytime, true);
    let batched = run_variant(
        "BatchedAnytime",
        &graph,
        &queries,
        &gt,
        &BatchedAnytime::default(),
        true,
    );

    println!();
    print_summary_table(&[&oneshot, &naive, &batched]);

    println!();
    println!("Time-to-Quality (median µs to reach fraction of final recall):");
    print_ttq_curve(&naive, "NaiveAnytime");
    print_ttq_curve(&batched, "BatchedAnytime");

    println!();
    println!("==== ACCEPTANCE CHECKS ====");
    check(
        "OneShot recall ≥ minimum",
        oneshot.recall >= MIN_BASELINE_RECALL,
        format!("{:.4} ≥ {MIN_BASELINE_RECALL}", oneshot.recall),
    );
    check(
        "NaiveAnytime final recall == OneShot",
        (naive.recall - oneshot.recall).abs() <= RECALL_EXACT_TOL,
        format!("Δ = {:.2e}", (naive.recall - oneshot.recall).abs()),
    );
    check(
        "BatchedAnytime final recall == OneShot",
        (batched.recall - oneshot.recall).abs() <= RECALL_EXACT_TOL,
        format!("Δ = {:.2e}", (batched.recall - oneshot.recall).abs()),
    );
    check(
        "Top-k monotonicity (NaiveAnytime)",
        naive.monotone_ok,
        "all consecutive snapshot pairs non-regressing".into(),
    );
    check(
        "Top-k monotonicity (BatchedAnytime)",
        batched.monotone_ok,
        "all consecutive snapshot pairs non-regressing".into(),
    );
    let reduction = naive.total_snapshots as f32 / batched.total_snapshots.max(1) as f32;
    check(
        "BatchedAnytime emits ≥ 4× fewer snapshots than NaiveAnytime",
        reduction >= MIN_BATCHED_REDUCTION,
        format!(
            "naive={} batched={} ratio={:.2}×",
            naive.total_snapshots, batched.total_snapshots, reduction
        ),
    );

    println!();
    println!("Build: {build_ms} ms | Queries: {N_QUERIES} | K={K} | EF={EF}");
    println!("(numbers above are real cargo-run output — re-run to regenerate)");
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-variant runner
// ──────────────────────────────────────────────────────────────────────────────

struct VariantStats {
    name: &'static str,
    recall: f32,
    lat: LatencyStats,
    total_snapshots: usize,
    monotone_ok: bool,
    /// For TTQ: per-query Vec<(elapsed_ns, recall_so_far)> snapshots.
    per_query_curves: Vec<Vec<(u128, f32)>>,
    #[allow(dead_code)]
    final_pops: f64,
    final_expansions: f64,
}

fn run_variant(
    name: &'static str,
    graph: &FlatGraph,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    searcher: &dyn Searcher,
    capture_curves: bool,
) -> VariantStats {
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut recalls = 0.0f32;
    let mut total_snaps = 0usize;
    let mut total_pops = 0u64;
    let mut total_exp = 0u64;
    let mut curves: Vec<Vec<(u128, f32)>> = Vec::with_capacity(if capture_curves {
        queries.len()
    } else {
        0
    });
    let mut monotone_ok = true;

    for (qi, q) in queries.iter().enumerate() {
        let mut snaps: Vec<AnytimeSnapshot> = vec![];
        let t = Instant::now();
        let r: SearchResult = searcher.search(graph, q, K, EF, ENTRY, &mut |s| {
            if capture_curves {
                snaps.push(s.clone());
            }
        });
        latencies.push(t.elapsed());

        // Final recall
        let pred: Vec<u32> = r.neighbors.iter().map(|x| x.0).collect();
        let rec = recall_at_k(&pred, &gt[qi], K);
        recalls += rec;
        total_snaps += r.snapshots_emitted;
        total_pops += r.pops as u64;
        total_exp += r.expansions as u64;

        if capture_curves {
            // Check monotonicity
            for w in snaps.windows(2) {
                let (a, b) = (&w[0], &w[1]);
                let best_a = a.neighbors.first().map(|x| x.1).unwrap_or(f32::INFINITY);
                let best_b = b.neighbors.first().map(|x| x.1).unwrap_or(f32::INFINITY);
                if best_b > best_a + 1e-7 {
                    monotone_ok = false;
                }
                if a.neighbors.len() == K && b.neighbors.len() == K {
                    let far_a = a.neighbors.last().unwrap().1;
                    let far_b = b.neighbors.last().unwrap().1;
                    if far_b > far_a + 1e-7 {
                        monotone_ok = false;
                    }
                }
            }
            // Build TTQ curve: snapshot recall vs elapsed_ns.
            let mut curve: Vec<(u128, f32)> = snaps
                .iter()
                .map(|s| {
                    let pred: Vec<u32> = s.neighbors.iter().map(|x| x.0).collect();
                    (s.elapsed_ns, recall_at_k(&pred, &gt[qi], K))
                })
                .collect();
            // Always include final result at full elapsed.
            curve.push((
                latencies.last().unwrap().as_nanos(),
                rec,
            ));
            curves.push(curve);
        }
    }

    VariantStats {
        name,
        recall: recalls / queries.len() as f32,
        lat: LatencyStats::from_durations(latencies),
        total_snapshots: total_snaps,
        monotone_ok,
        per_query_curves: curves,
        final_pops: total_pops as f64 / queries.len() as f64,
        final_expansions: total_exp as f64 / queries.len() as f64,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Output helpers
// ──────────────────────────────────────────────────────────────────────────────

fn print_header() {
    println!("================================================================");
    println!("  Anytime HNSW — Progressive Top-K with Monotone Guarantees");
    println!("================================================================");
}

fn print_summary_table(stats: &[&VariantStats]) {
    println!(
        "{:<18} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "recall@k", "p50 µs", "p95 µs", "p99 µs", "snaps/q", "exp/q"
    );
    println!("{:─<82}", "");
    for s in stats {
        let snaps_per_q = s.total_snapshots as f64 / N_QUERIES as f64;
        println!(
            "{:<18} {:>10.4} {:>12.1} {:>10.1} {:>10.1} {:>10.2} {:>10.1}",
            s.name,
            s.recall,
            s.lat.p50_us,
            s.lat.p95_us,
            s.lat.p99_us,
            snaps_per_q,
            s.final_expansions,
        );
    }
}

fn print_ttq_curve(stats: &VariantStats, name: &str) {
    if stats.per_query_curves.is_empty() {
        return;
    }
    // For each fraction f, compute per-query first elapsed_ns where snapshot
    // recall ≥ f × final_recall_of_that_query, then report median.
    let mut row = format!("  {:<18}", name);
    for &frac in TTQ_FRACTIONS {
        let mut times_us: Vec<f64> = Vec::with_capacity(stats.per_query_curves.len());
        for curve in &stats.per_query_curves {
            let final_r = curve.last().map(|x| x.1).unwrap_or(0.0);
            let target = frac * final_r;
            let mut hit_us: Option<f64> = None;
            for &(ns, r) in curve {
                if r + 1e-7 >= target {
                    hit_us = Some(ns as f64 / 1_000.0);
                    break;
                }
            }
            if let Some(us) = hit_us {
                times_us.push(us);
            }
        }
        times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if times_us.is_empty() {
            f64::NAN
        } else {
            times_us[times_us.len() / 2]
        };
        row.push_str(&format!(" {:>3.0}%:{:>7.1}µs", frac * 100.0, med));
    }
    println!("{row}");
}

fn check(name: &str, ok: bool, detail: String) {
    println!(
        "  [{}] {name} — {detail}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
}
