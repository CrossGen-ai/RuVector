//! `cargo run --release --bin benchmark` — produces the numbers
//! quoted in docs/research/nightly/2026-06-22-nndescent-bulk-hnsw.

use std::time::Instant;

use ruvector_nndescent::{
    BruteForceBuilder, Dataset, DistanceCounter, KnnGraph, KnnGraphBuilder, NnDescentBuilder,
    NnDescentConfig, SeededHnsw, SeededHnswConfig,
};

#[derive(Debug)]
struct Variant {
    name: &'static str,
    build_ms: f64,
    distance_ops: u64,
    graph_recall: f64,
    search_recall: f64,
    search_qps: f64,
}

fn search_eval(vectors: &[Vec<f32>], graph: &KnnGraph, queries: &[Vec<f32>], k: usize, truth_search: &[Vec<u32>]) -> (f64, f64) {
    let n_entries = ((vectors.len() as f64).log2() as usize).max(8);
    let hnsw = SeededHnsw::new(vectors, graph, SeededHnswConfig {
        ef_search: 64,
        entry_points: SeededHnswConfig::spread_entries(vectors.len(), n_entries),
    });
    let t0 = Instant::now();
    let mut hits = 0u64;
    let mut total = 0u64;
    for (qi, q) in queries.iter().enumerate() {
        let got = hnsw.search(q, k);
        let truth: std::collections::HashSet<u32> = truth_search[qi].iter().copied().collect();
        total += truth.len() as u64;
        for g in got { if truth.contains(&g) { hits += 1; } }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    let recall = if total == 0 { 0.0 } else { hits as f64 / total as f64 };
    (recall, qps)
}

fn brute_query_truth(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    use ruvector_nndescent::l2_sq;
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let mut all: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(q, v), i as u32))
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.push(all.into_iter().take(k).map(|(_, id)| id).collect());
    }
    out
}

fn main() {
    // Dataset — chosen so brute force is still tractable (<30s)
    // but N is large enough to expose NN-Descent's sub-quadratic scaling.
    let n = 8_000;
    let dim = 64;
    let k_clusters = 40;
    let n_queries = 300;
    let k = 16;
    let seed = 42;

    println!("=== ruvector-nndescent: bulk kNN graph construction ===");
    println!("Dataset: N={n}, dim={dim}, clusters={k_clusters}, queries={n_queries}, k={k}, seed={seed}\n");

    let ds = Dataset::synthetic_clustered(n, dim, k_clusters, n_queries, seed);

    // Ground truth: brute force graph + brute force query truth
    println!("Building ground truth (brute force)...");
    let bf_counter = DistanceCounter::new();
    let t0 = Instant::now();
    let bf_graph = BruteForceBuilder.build(&ds.vectors, k, &bf_counter);
    let bf_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let bf_ops = bf_counter.count();
    println!("  brute force: {bf_build_ms:.1} ms, {bf_ops} distance ops");

    let query_truth = brute_query_truth(&ds.vectors, &ds.queries, k);
    let (bf_recall_search, bf_qps) = search_eval(&ds.vectors, &bf_graph, &ds.queries, k, &query_truth);

    let mut results: Vec<Variant> = Vec::new();
    results.push(Variant {
        name: "BruteForce",
        build_ms: bf_build_ms,
        distance_ops: bf_ops,
        graph_recall: 1.000,
        search_recall: bf_recall_search,
        search_qps: bf_qps,
    });

    // NN-Descent basic
    {
        let cfg = NnDescentConfig { rho: 0.5, delta: 1e-3, max_iters: 12, use_reverse: false, seed: 1 };
        let counter = DistanceCounter::new();
        let t0 = Instant::now();
        let g = NnDescentBuilder::new(cfg.clone()).build(&ds.vectors, k, &counter);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ops = counter.count();
        let recall = g.recall_against(&bf_graph, k);
        let (sr, qps) = search_eval(&ds.vectors, &g, &ds.queries, k, &query_truth);
        results.push(Variant {
            name: "NnDescent-Basic",
            build_ms: ms,
            distance_ops: ops,
            graph_recall: recall,
            search_recall: sr,
            search_qps: qps,
        });
    }

    // NN-Descent local-join (with reverse)
    {
        let cfg = NnDescentConfig { rho: 0.5, delta: 1e-3, max_iters: 12, use_reverse: true, seed: 2 };
        let counter = DistanceCounter::new();
        let t0 = Instant::now();
        let g = NnDescentBuilder::new(cfg).build(&ds.vectors, k, &counter);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ops = counter.count();
        let recall = g.recall_against(&bf_graph, k);
        let (sr, qps) = search_eval(&ds.vectors, &g, &ds.queries, k, &query_truth);
        results.push(Variant {
            name: "NnDescent-LocalJoin",
            build_ms: ms,
            distance_ops: ops,
            graph_recall: recall,
            search_recall: sr,
            search_qps: qps,
        });
    }

    // ---- Report ----
    println!("\n=== Results ===");
    println!(
        "{:<22} {:>12} {:>14} {:>12} {:>12} {:>10}",
        "Variant", "Build (ms)", "Dist ops", "Graph rec", "Srch rec", "Srch QPS"
    );
    println!("{}", "-".repeat(86));
    for v in &results {
        println!(
            "{:<22} {:>12.1} {:>14} {:>12.3} {:>12.3} {:>10.0}",
            v.name, v.build_ms, v.distance_ops, v.graph_recall, v.search_recall, v.search_qps
        );
    }

    // Speedup deltas
    let bf = &results[0];
    println!("\n=== Speedup over BruteForce ===");
    for v in results.iter().skip(1) {
        let t_speedup = bf.build_ms / v.build_ms;
        let d_ratio = (v.distance_ops as f64) / (bf.distance_ops as f64);
        println!(
            "{:<22} time={:>5.2}x faster | distance-ops={:>5.1}% of brute | graph recall={:.3}",
            v.name, t_speedup, d_ratio * 100.0, v.graph_recall
        );
    }
    println!("\nDone.");
}
