//! symphony-qg-bench — head-to-head benchmark of 3 variants:
//!
//!   A. Flat brute force         (oracle, ground truth, slowest)
//!   B. Graph-only (f32)         (kNN graph traversal, full f32 distances)
//!   C. SymphonyQG (this work)   (graph traversal on Hamming + f32 rerank)
//!
//! Measures: build time, query latency (median + p95), recall@10 vs A.
//! Memory footprint (estimated) is also printed.
//!
//! Run: `cargo run --release -p ruvector-symphony-qg --bin symphony-qg-bench`

use ruvector_symphony_qg::{
    brute_force_knn, recall_at_k,
    graph::{GraphParams, KnnGraph},
    symphony::{IndexParams, SymphonyIndex},
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

const N: usize = 20_000;
const D: usize = 128;
const NQ: usize = 200;
const TOPK: usize = 10;
const SEED_DATA: u64 = 0xBEEF;
const SEED_QUERY: u64 = 0xCAFE;

fn gen_unit(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| {
        let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
        v.iter().map(|x| x / n).collect()
    }).collect()
}

fn percentile(v: &mut Vec<u128>, p: f64) -> u128 {
    v.sort_unstable();
    let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[idx]
}

fn main() {
    println!("# ruvector-symphony-qg bench  n={N} d={D} nq={NQ} topk={TOPK}");

    println!("\n## Generating data ...");
    let t = Instant::now();
    let corpus = gen_unit(N, D, SEED_DATA);
    let queries = gen_unit(NQ, D, SEED_QUERY);
    println!("  data gen: {:?}", t.elapsed());

    // ------ A. Brute force ground truth ------
    println!("\n## [A] Brute force (oracle)");
    let t = Instant::now();
    let mut bf_lat = Vec::with_capacity(NQ);
    let mut truth = Vec::with_capacity(NQ);
    for q in &queries {
        let qt = Instant::now();
        let h = brute_force_knn(&corpus, q, TOPK);
        bf_lat.push(qt.elapsed().as_nanos());
        truth.push(h);
    }
    println!("  total search: {:?}", t.elapsed());
    print_lat("A brute", &mut bf_lat);

    // ------ B. Graph-only (f32) ------
    println!("\n## [B] Graph-only (f32)");
    let gp = GraphParams { k: 32, iters: 4, ef_search: 96, seed: 0xAA };
    let t = Instant::now();
    let g = KnnGraph::build(&corpus, gp.clone());
    println!("  build: {:?}", t.elapsed());
    let g_bytes = g.adj.len() * std::mem::size_of::<u32>() + N * D * std::mem::size_of::<f32>();
    println!("  est mem: {:.2} MB", g_bytes as f64 / (1024.0*1024.0));

    let mut g_lat = Vec::with_capacity(NQ);
    let mut g_recall = 0.0f32;
    for (i, q) in queries.iter().enumerate() {
        let qt = Instant::now();
        let hits = g.search_f32(&corpus, q, TOPK, gp.ef_search, 0xBB);
        g_lat.push(qt.elapsed().as_nanos());
        g_recall += recall_at_k(&hits, &truth[i]);
    }
    g_recall /= NQ as f32;
    print_lat("B graph", &mut g_lat);
    println!("  recall@{}: {:.3}", TOPK, g_recall);

    // ------ C. SymphonyQG ------
    println!("\n## [C] SymphonyQG (Hamming traversal + f32 rerank)");
    let params = IndexParams {
        graph: gp.clone(),
        quant_seed: 0xC0DE,
        rerank: 64,
    };
    let t = Instant::now();
    let idx = SymphonyIndex::build(&corpus, params.clone());
    println!("  build: {:?}", t.elapsed());
    println!("  est mem: {:.2} MB (incl {:.2} MB binary codes)",
        idx.estimated_bytes() as f64 / (1024.0*1024.0),
        (idx.codes.len() * 8) as f64 / (1024.0*1024.0));

    let mut s_lat = Vec::with_capacity(NQ);
    let mut s_recall = 0.0f32;
    let mut total_visited = 0u64;
    let mut total_ham = 0u64;
    let mut total_f32 = 0u64;
    for (i, q) in queries.iter().enumerate() {
        let qt = Instant::now();
        let (hits, stats) = idx.search(q, TOPK);
        s_lat.push(qt.elapsed().as_nanos());
        s_recall += recall_at_k(&hits, &truth[i]);
        total_visited += stats.visited as u64;
        total_ham += stats.ham_ops as u64;
        total_f32 += stats.f32_ops as u64;
    }
    s_recall /= NQ as f32;
    print_lat("C symphony", &mut s_lat);
    println!("  recall@{}: {:.3}", TOPK, s_recall);
    println!("  avg per-query: visited={}, ham_ops={}, f32_ops={}",
        total_visited / NQ as u64,
        total_ham / NQ as u64,
        total_f32 / NQ as u64);

    // ------ Summary table ------
    println!("\n## Summary");
    let med_a = median(&mut bf_lat) as f64 / 1000.0;
    let med_b = median(&mut g_lat) as f64 / 1000.0;
    let med_c = median(&mut s_lat) as f64 / 1000.0;
    println!("{:<14} {:>10} {:>10} {:>10}", "variant", "med_us", "p95_us", "recall@10");
    println!("{:<14} {:>10.1} {:>10.1} {:>10}", "A brute",     med_a, percentile(&mut bf_lat, 0.95) as f64 / 1000.0, "1.000");
    println!("{:<14} {:>10.1} {:>10.1} {:>10.3}", "B graph-f32", med_b, percentile(&mut g_lat, 0.95) as f64 / 1000.0, g_recall);
    println!("{:<14} {:>10.1} {:>10.1} {:>10.3}", "C symphony",   med_c, percentile(&mut s_lat, 0.95) as f64 / 1000.0, s_recall);

    if med_c < med_b {
        println!("\nSymphonyQG faster than graph-f32 by {:.2}x", med_b / med_c);
    } else {
        println!("\nSymphonyQG slower than graph-f32 by {:.2}x (recall trade is {:.3} vs {:.3})",
            med_c / med_b, s_recall, g_recall);
    }
}

fn median(v: &mut Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn print_lat(label: &str, lat: &mut Vec<u128>) {
    let med = median(lat);
    let p95 = percentile(lat, 0.95);
    let p99 = percentile(lat, 0.99);
    println!("  {}: median={:.1}us p95={:.1}us p99={:.1}us",
        label,
        med as f64 / 1000.0,
        p95 as f64 / 1000.0,
        p99 as f64 / 1000.0);
}
