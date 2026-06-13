//! deg-demo: runs BruteForce, KnnGraph, and DEG over a synthetic
//! Gaussian-mixture dataset and reports recall@10 + queries-per-second.

use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::Instant;

use ruvector_deg::{AnnIndex, BruteForce, Deg, KnnGraph, recall_at_k};

fn gen_dataset(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..clusters)];
            (0..dim).map(|j| c[j] + rng.gen_range(-0.5..0.5)).collect()
        })
        .collect()
}

fn run<I: AnnIndex>(name: &str, idx: &mut I, queries: &[Vec<f32>], truth: &[Vec<(usize, f32)>], k: usize) {
    let t_q = Instant::now();
    let mut sum_recall = 0.0f32;
    for (qi, q) in queries.iter().enumerate() {
        let r = idx.search(q, k);
        sum_recall += recall_at_k(&r, &truth[qi], k);
    }
    let qsec = queries.len() as f32 / t_q.elapsed().as_secs_f32();
    println!(
        "{:<14} n={:<6} mem={:>10} B  recall@{}={:.4}  qps={:>10.1}",
        name,
        idx.len(),
        idx.mem_bytes(),
        k,
        sum_recall / queries.len() as f32,
        qsec,
    );
}

fn main() {
    let n: usize = 5_000;
    let dim: usize = 64;
    let n_queries: usize = 200;
    let k: usize = 10;

    println!("== ruvector-deg demo  (n={}, dim={}, queries={}, k={}) ==", n, dim, n_queries, k);

    let data = gen_dataset(n, dim, 20, 42);
    let queries = gen_dataset(n_queries, dim, 20, 1337);

    // Ground truth from brute-force.
    let mut gt = BruteForce::new(dim);
    for v in &data { gt.insert(v.clone()); }
    let truth: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| gt.search(q, k)).collect();

    println!("\n-- BruteForce (exact baseline) --");
    let mut bf = BruteForce::new(dim);
    let t = Instant::now();
    for v in &data { bf.insert(v.clone()); }
    println!("build: {:.2?}", t.elapsed());
    run("BruteForce", &mut bf, &queries, &truth, k);

    println!("\n-- KnnGraph (static k-NN, beam search) --");
    let mut kg = KnnGraph::new(dim, 24, 64);
    let t = Instant::now();
    for v in &data { kg.insert(v.clone()); }
    kg.build();
    println!("build: {:.2?}", t.elapsed());
    run("KnnGraph", &mut kg, &queries, &truth, k);

    println!("\n-- DEG (dynamic, RNG pruning) --");
    let mut deg = Deg::new(dim, 24, 64, 64);
    let t = Instant::now();
    for v in &data { deg.insert(v.clone()); }
    println!("build: {:.2?}", t.elapsed());
    run("DEG", &mut deg, &queries, &truth, k);

    println!("\n-- DEG (no RNG pruning, M=24) --");
    let mut deg2 = Deg::new(dim, 24, 64, 64);
    deg2.rng_pruning = false;
    let t = Instant::now();
    for v in &data { deg2.insert(v.clone()); }
    println!("build: {:.2?}", t.elapsed());
    run("DEG-noRNG", &mut deg2, &queries, &truth, k);

    // ---- Pareto sweep: ef_search vs recall vs QPS on the RNG-pruned graph ----
    println!("\n-- DEG ef_search sweep (RNG pruning on) --");
    println!("{:<6} {:<10} {:<10}", "ef", "recall@10", "qps");
    for &ef in &[32usize, 64, 128, 256, 512] {
        deg.ef_search = ef;
        deg.n_entries = 16;
        let t_q = Instant::now();
        let mut sum_recall = 0.0f32;
        for (qi, q) in queries.iter().enumerate() {
            let r = deg.search(q, k);
            sum_recall += ruvector_deg::recall_at_k(&r, &truth[qi], k);
        }
        let qsec = queries.len() as f32 / t_q.elapsed().as_secs_f32();
        println!("{:<6} {:<10.4} {:<10.1}", ef, sum_recall / queries.len() as f32, qsec);
    }
}
