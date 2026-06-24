//! Demo binary: builds three NSW variants and prints real benchmark numbers.
//!
//! Run: `cargo run --release -p ruvector-hub-hnsw`

use std::time::Instant;

use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_hub_hnsw::{
    hubness, AnnIndex, BaselineNsw, HubNsw, IndegreeCap, NswParams, Vector,
};

fn synth_dataset(n: usize, d: usize, seed: u64) -> Vec<Vector> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
        .collect()
}

fn brute_top_k(vectors: &[Vector], q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (ruvector_hub_hnsw::l2_sq(v, q), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

#[derive(serde::Serialize)]
struct VariantReport {
    variant: String,
    build_ms: u128,
    mean_us: f64,
    p95_us: f64,
    qps: f64,
    recall_at_10: f64,
    indeg_mean: f32,
    indeg_max: usize,
    indeg_p99: usize,
    indeg_gini: f32,
    hub_fraction: f32,
    edge_count: usize,
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() as f64) * p) as usize;
    *xs.get(idx.min(xs.len().saturating_sub(1))).unwrap_or(&0.0)
}

fn evaluate(
    label: &str,
    build_ms: u128,
    idx: &dyn AnnIndex,
    queries: &[Vector],
    truth: &[Vec<u32>],
    k: usize,
    ef: usize,
) -> VariantReport {
    let n_q = queries.len();
    let mut lat_us = Vec::with_capacity(n_q);
    let mut hits = 0usize;
    let total = n_q * k;
    for (qi, q) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let res = idx.search(q, k, ef);
        let dt = t0.elapsed().as_secs_f64() * 1.0e6;
        lat_us.push(dt);
        for (rid, _) in &res {
            if truth[qi].contains(rid) {
                hits += 1;
            }
        }
    }
    let mean = lat_us.iter().sum::<f64>() / n_q as f64;
    let p95 = percentile(lat_us.clone(), 0.95);
    let qps = 1.0e6 / mean;
    let stats = hubness::stats_from(idx.adjacency());
    let edge_count: usize = idx.adjacency().iter().map(|e| e.len()).sum();
    VariantReport {
        variant: label.to_string(),
        build_ms,
        mean_us: mean,
        p95_us: p95,
        qps,
        recall_at_10: hits as f64 / total as f64,
        indeg_mean: stats.mean,
        indeg_max: stats.max,
        indeg_p99: stats.p99,
        indeg_gini: stats.gini,
        hub_fraction: stats.hub_fraction,
        edge_count,
    }
}

fn main() {
    let n: usize = 5_000;
    let d: usize = 64;
    let n_q: usize = 200;
    let k: usize = 10;
    let ef: usize = 64;
    let params = NswParams {
        m: 16,
        ef_construction: 64,
        seed_entry: 0,
    };

    println!("=== Hubness-Aware HNSW: PoC benchmark ===");
    println!(
        "dataset: N={} D={} queries={} k={} ef={} M={} efC={}",
        n, d, n_q, k, ef, params.m, params.ef_construction
    );

    let base_vecs = synth_dataset(n, d, 42);
    let queries = synth_dataset(n_q, d, 99);

    // Ground truth via brute force.
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| brute_top_k(&base_vecs, q, k)).collect();
    println!(
        "[brute-force ground truth] {} queries in {:.2}s",
        n_q,
        t0.elapsed().as_secs_f64()
    );

    // Variant 1: baseline.
    let t0 = Instant::now();
    let base = BaselineNsw::build(base_vecs.clone(), params.clone());
    let build_base = t0.elapsed().as_millis();
    let r_base = evaluate("BaselineNsw", build_base, &base, &queries, &truth, k, ef);

    // Variant 2: light anti-hub.
    let t0 = Instant::now();
    let light = HubNsw::build(base_vecs.clone(), params.clone(), IndegreeCap::Light);
    let build_light = t0.elapsed().as_millis();
    let r_light = evaluate("HubNsw[Light]", build_light, &light, &queries, &truth, k, ef);

    // Variant 3: aggressive anti-hub.
    let t0 = Instant::now();
    let agg = HubNsw::build(base_vecs, params, IndegreeCap::Aggressive);
    let build_agg = t0.elapsed().as_millis();
    let r_agg = evaluate("HubNsw[Aggressive]", build_agg, &agg, &queries, &truth, k, ef);

    println!();
    println!(
        "{:<22}  {:>9}  {:>10}  {:>10}  {:>9}  {:>10}  {:>9}  {:>8}  {:>8}  {:>10}  {:>10}",
        "variant",
        "build_ms",
        "mean_us",
        "p95_us",
        "qps",
        "recall@10",
        "indeg_mu",
        "indeg_mx",
        "indeg_p99",
        "indeg_gini",
        "hub_frac"
    );
    for r in [&r_base, &r_light, &r_agg] {
        println!(
            "{:<22}  {:>9}  {:>10.2}  {:>10.2}  {:>9.0}  {:>10.4}  {:>9.2}  {:>8}  {:>8}  {:>10.4}  {:>10.4}",
            r.variant,
            r.build_ms,
            r.mean_us,
            r.p95_us,
            r.qps,
            r.recall_at_10,
            r.indeg_mean,
            r.indeg_max,
            r.indeg_p99,
            r.indeg_gini,
            r.hub_fraction
        );
    }

    let out = serde_json::to_string_pretty(&serde_json::json!({
        "config": { "n": n, "d": d, "n_q": n_q, "k": k, "ef": ef, "m": 16, "ef_construction": 64 },
        "results": [r_base, r_light, r_agg]
    }))
    .unwrap();
    println!("\n--- JSON ---\n{}", out);
}
