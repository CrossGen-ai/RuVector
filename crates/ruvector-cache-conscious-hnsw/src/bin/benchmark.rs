//! Cache-conscious HNSW benchmark: 3 orderings x measured latency + recall.
//!
//! Produces real numbers (not aspirational). Usage:
//!   cargo run --release -p ruvector-cache-conscious-hnsw --bin benchmark
//!   cargo run --release -p ruvector-cache-conscious-hnsw --bin benchmark -- \
//!       --n 20000 --dim 96 --queries 500 --degree 32 --ef 64 --k 10

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_cache_conscious_hnsw::{
    brute_topk, build_graph, mean_edge_span, recall_at_k, reorder_with, search,
    reorder::{Bfs, ReverseCuthillMcKee},
};
use std::time::Instant;

struct Args {
    n: usize, dim: usize, queries: usize,
    degree: usize, pool: usize, ef: usize, k: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 20_000, dim: 96, queries: 500,
        degree: 32, pool: 128, ef: 64, k: 10,
        seed: 42,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next();
        match (k.as_str(), v.as_deref()) {
            ("--n", Some(x)) => a.n = x.parse().unwrap(),
            ("--dim", Some(x)) => a.dim = x.parse().unwrap(),
            ("--queries", Some(x)) => a.queries = x.parse().unwrap(),
            ("--degree", Some(x)) => a.degree = x.parse().unwrap(),
            ("--pool", Some(x)) => a.pool = x.parse().unwrap(),
            ("--ef", Some(x)) => a.ef = x.parse().unwrap(),
            ("--k", Some(x)) => a.k = x.parse().unwrap(),
            ("--seed", Some(x)) => a.seed = x.parse().unwrap(),
            _ => {}
        }
    }
    a
}

/// Clustered mixture-of-Gaussians vectors — a realistic proxy for the
/// clustered structure of real embedding corpora (BERT/OpenAI/etc).
fn gen_clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = 64.min(n / 32).max(4);
    // draw cluster centers on a sphere-ish shell
    let center_n = Normal::new(0.0f32, 3.0f32).unwrap();
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| center_n.sample(&mut rng)).collect())
        .collect();
    let jitter = Normal::new(0.0f32, 0.4f32).unwrap();
    let mut v = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % n_clusters];
        for d in 0..dim { v.push(c[d] + jitter.sample(&mut rng)); }
    }
    v
}

fn gen_gaussian(n: usize, dim: usize, seed: u64) -> Vec<f32> { gen_clustered(n, dim, seed) }

fn gen_queries(n_q: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    (0..n_q).map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect()).collect()
}

fn bench_variant(
    label: &str,
    graph: &ruvector_cache_conscious_hnsw::FlatGraph,
    queries: &[Vec<f32>],
    _truth: &[Vec<u32>],
    ef: usize, k: usize,
) -> (f64, f32, f64) {
    // Per-variant truth: brute-force on THIS graph's vectors so IDs match
    // (reordering renames IDs — a shared truth table would be nonsense).
    let per_truth: Vec<Vec<u32>> = queries.iter()
        .map(|q| brute_topk(&graph.vectors, graph.dim, q, k))
        .collect();

    // Warmup
    for q in queries.iter().take(10) { let _ = search(graph, q, k, ef); }

    let mut approx: Vec<Vec<(f32, u32)>> = Vec::with_capacity(queries.len());
    let t0 = Instant::now();
    for q in queries { approx.push(search(graph, q, k, ef)); }
    let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
    let per_query_us = elapsed_us / queries.len() as f64;

    let rec = recall_at_k(&approx, &per_truth, k);
    let span = mean_edge_span(graph);
    println!(
        "  {:24} qps={:>8.0}  avg_us={:>7.2}  recall@{}={:.3}  mean_edge_span={:.1}",
        label,
        queries.len() as f64 / (elapsed_us / 1e6),
        per_query_us, k, rec, span,
    );
    (per_query_us, rec, span)
}

fn main() {
    let a = parse_args();
    println!("== ruvector-cache-conscious-hnsw benchmark ==");
    println!("n={}  dim={}  queries={}  degree={}  pool={}  ef={}  k={}  seed={}",
        a.n, a.dim, a.queries, a.degree, a.pool, a.ef, a.k, a.seed);

    let t = Instant::now();
    let vectors = gen_gaussian(a.n, a.dim, a.seed);
    println!("gen_vectors: {:.2}s  ({:.1} MB)",
        t.elapsed().as_secs_f64(),
        (a.n * a.dim * 4) as f64 / 1e6);

    let t = Instant::now();
    let g = build_graph(vectors.clone(), a.dim, a.degree, a.pool, a.seed ^ 0x9E37);
    println!("build_graph: {:.2}s  (max_degree={}, entry={})",
        t.elapsed().as_secs_f64(), g.max_degree, g.entry);

    let t = Instant::now();
    let queries = gen_queries(a.queries, a.dim, a.seed.wrapping_add(1));
    let truth: Vec<Vec<u32>> = queries.iter()
        .map(|q| brute_topk(&vectors, a.dim, q, a.k))
        .collect();
    println!("brute-force truth: {:.2}s", t.elapsed().as_secs_f64());

    println!("\n-- Variants --");
    let base = bench_variant("insertion (baseline)", &g, &queries, &truth, a.ef, a.k);

    let g_bfs = {
        let ord = Bfs;
        let t = Instant::now();
        let out = reorder_with(&g, &ord);
        println!("  [reorder bfs           ] {:.3}s", t.elapsed().as_secs_f64());
        out
    };
    let bfs = bench_variant("bfs", &g_bfs, &queries, &truth, a.ef, a.k);

    let g_rcm = {
        let ord = ReverseCuthillMcKee;
        let t = Instant::now();
        let out = reorder_with(&g, &ord);
        println!("  [reorder rcm           ] {:.3}s", t.elapsed().as_secs_f64());
        out
    };
    let rcm = bench_variant("reverse-cuthill-mckee", &g_rcm, &queries, &truth, a.ef, a.k);

    println!("\n== Summary (speedup = baseline_us / variant_us) ==");
    println!("  baseline (insertion)  : {:.2} us/q  recall={:.3}  span={:.1}", base.0, base.1, base.2);
    println!("  bfs                   : {:.2} us/q  recall={:.3}  span={:.1}  speedup={:.2}x  span_reduction={:.2}x",
        bfs.0, bfs.1, bfs.2, base.0 / bfs.0, base.2 / bfs.2);
    println!("  reverse-cuthill-mckee : {:.2} us/q  recall={:.3}  span={:.1}  speedup={:.2}x  span_reduction={:.2}x",
        rcm.0, rcm.1, rcm.2, base.0 / rcm.0, base.2 / rcm.2);

    // Acceptance check: recall must be preserved (within 1%) across orderings.
    let recall_ok = (bfs.1 - base.1).abs() < 0.01 && (rcm.1 - base.1).abs() < 0.01;
    println!("\nacceptance (recall preserved within 1%): {}", if recall_ok { "PASS" } else { "GAP" });
    if !recall_ok {
        println!("  note: reordering must be a pure isomorphism; recall drift indicates a bug.");
    }
}
