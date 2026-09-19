//! Real benchmark binary — measures build time, insert throughput, query
//! recall@10, and QPS for the three index variants over a clustered synthetic
//! dataset. All numbers reported are wall-clock from `std::time::Instant`.

use std::time::Instant;

use ruvector_deg_frozen_backbone::{
    brute_force_topk, make_clustered_dataset, recall_at_k, AnnIndex, FrozenBackboneDeg, HnswLite,
    MutableDeg, VecStore,
};

struct Row {
    name: String,
    build_ms: f64,
    insert_us_p50: f64,
    recall_at_10: f64,
    qps: f64,
    edges: usize,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_index(name: &str, mut idx: Box<dyn AnnIndex>, data: &[Vec<f32>], queries: &[Vec<f32>], truth: &[Vec<(u32, f32)>]) -> Row {
    let mut insert_us: Vec<f64> = Vec::with_capacity(data.len());
    let t0 = Instant::now();
    for (i, v) in data.iter().enumerate() {
        let s = Instant::now();
        idx.insert(i as u32, v.clone());
        insert_us.push(s.elapsed().as_micros() as f64);
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let edges = idx.n_edges();

    // Query phase.
    let ef_search = 64;
    let q0 = Instant::now();
    let mut recall_sum = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let res = idx.search(q, 10, ef_search);
        recall_sum += recall_at_k(&res, &truth[qi], 10) as f64;
    }
    let q_secs = q0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / q_secs;
    let recall_at_10 = recall_sum / queries.len() as f64;

    Row {
        name: name.to_string(),
        build_ms,
        insert_us_p50: median(&mut insert_us),
        recall_at_10,
        qps,
        edges,
    }
}

fn main() {
    let n = std::env::var("DEG_N").ok().and_then(|s| s.parse().ok()).unwrap_or(5000);
    let dim = std::env::var("DEG_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_q = 200usize;
    println!("=== ruvector-deg-frozen-backbone bench ===");
    println!("n={n}  dim={dim}  n_queries={n_q}  M=16  ef_build=64  ef_search=64  k=10");

    let all = make_clustered_dataset(n + n_q, dim, 16, 20260919);
    let (data, queries) = all.split_at(n);
    let data = data.to_vec();
    let queries = queries.to_vec();

    // Ground truth via brute force.
    let mut store = VecStore::new(dim);
    for v in &data {
        store.push(v);
    }
    let mut truth = Vec::with_capacity(queries.len());
    let tt = Instant::now();
    for q in &queries {
        truth.push(brute_force_topk(&store, q, 10));
    }
    println!("brute-force truth built in {:.1} ms", tt.elapsed().as_secs_f64() * 1000.0);

    // Variants (three, per topic-agnostic spec: baseline + 2 alternatives).
    let variants: Vec<(&str, Box<dyn AnnIndex>)> = vec![
        ("hnsw-lite (baseline)", Box::new(HnswLite::new(dim, 16, 64))),
        ("deg-mutable", Box::new(MutableDeg::new(dim, 16, 64))),
        ("deg-frozen-backbone(0.25)", Box::new(FrozenBackboneDeg::new(dim, 16, 64, 0.25))),
    ];

    let mut rows = Vec::new();
    for (name, idx) in variants {
        let r = bench_index(name, idx, &data, &queries, &truth);
        rows.push(r);
    }

    // Also run frozen with 0.5 to expose sensitivity.
    let extra: Box<dyn AnnIndex> = Box::new(FrozenBackboneDeg::new(dim, 16, 64, 0.5));
    let r = bench_index("deg-frozen-backbone(0.50)", extra, &data, &queries, &truth);
    rows.push(r);

    println!();
    println!(
        "{:<32}  {:>10}  {:>12}  {:>12}  {:>10}  {:>10}",
        "variant", "build_ms", "insert_us_p50", "recall@10", "QPS", "edges"
    );
    for r in &rows {
        println!(
            "{:<32}  {:>10.1}  {:>12.2}  {:>12.4}  {:>10.1}  {:>10}",
            r.name, r.build_ms, r.insert_us_p50, r.recall_at_10, r.qps, r.edges
        );
    }

    // Emit machine-readable trailer for the research doc capture.
    println!("---BENCH-JSON-START---");
    print!("[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"build_ms\":{:.3},\"insert_us_p50\":{:.4},\"recall_at_10\":{:.6},\"qps\":{:.3},\"edges\":{}}}",
            r.name, r.build_ms, r.insert_us_p50, r.recall_at_10, r.qps, r.edges
        );
    }
    println!("]");
    println!("---BENCH-JSON-END---");
}
