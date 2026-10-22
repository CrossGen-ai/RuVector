//! Benchmark driver for SOAR vs RandomSpill vs PrimaryOnly on a synthetic
//! Gaussian-mixture dataset.
//!
//! Usage: `cargo run --release --example bench -p ruvector-soar-ivf`

use std::time::Instant;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ruvector_soar_ivf::*;

fn gen_mixture(n: usize, dim: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..k)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect())
        .collect();
    (0..n).map(|_| {
        let c = &centers[rng.gen_range(0..k)];
        c.iter().map(|m| m + (rng.gen::<f32>() - 0.5) * 2.0).collect()
    }).collect()
}

fn eval(
    idx: &IvfIndex,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
    nprobes: &[usize],
) -> Vec<(usize, f32, f64)> {
    // (nprobe, mean recall@k, mean us/query)
    let mut out = Vec::new();
    for &np in nprobes {
        let mut total_r = 0.0f32;
        let t0 = Instant::now();
        for (q, truth) in queries.iter().zip(truths.iter()) {
            let pred: Vec<u32> = idx.search(q, k, np).into_iter().map(|(i, _)| i).collect();
            total_r += recall_at_k(&pred, truth);
        }
        let elapsed = t0.elapsed();
        let us = elapsed.as_micros() as f64 / queries.len() as f64;
        out.push((np, total_r / queries.len() as f32, us));
    }
    out
}

fn main() {
    // --- Config ---
    let n = 50_000;
    let dim = 64;
    let n_clusters_data = 128;   // true clusters in synthetic data
    let n_centroids = 256;       // IVF centroids
    let kmeans_iters = 15;
    let n_queries = 500;
    let k = 10;
    let nprobes = [1usize, 2, 4, 8, 16, 32];

    println!("# ruvector-soar-ivf benchmark");
    println!("n={}, dim={}, centroids={}, queries={}, k={}", n, dim, n_centroids, n_queries, k);
    println!();

    // --- Data + ground truth ---
    let t0 = Instant::now();
    let data = gen_mixture(n + n_queries, dim, n_clusters_data, 2026_10_22);
    let queries: Vec<Vec<f32>> = data[n..].to_vec();
    let data: Vec<Vec<f32>> = data[..n].to_vec();
    println!("Generated data in {:.1}s", t0.elapsed().as_secs_f32());

    let t0 = Instant::now();
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| brute_topk(&data, q, k)).collect();
    println!("Built ground truth (brute force) in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Centroids (shared across all three variants for fairness) ---
    let t0 = Instant::now();
    let centroids = kmeans(&data, n_centroids, kmeans_iters, 1234);
    println!("Ran k-means ({} iters) in {:.1}s", kmeans_iters, t0.elapsed().as_secs_f32());
    println!();

    // --- Build each index ---
    let variants: Vec<(&str, Box<dyn IvfAssigner>)> = vec![
        ("primary-only", Box::new(PrimaryOnly)),
        ("random-spill", Box::new(RandomSpill { seed: 0xC0FFEE })),
        ("soar(l=0.5)",  Box::new(Soar { lambda: 0.5 })),
        ("soar(l=1.0)",  Box::new(Soar { lambda: 1.0 })),
        ("soar(l=2.0)",  Box::new(Soar { lambda: 2.0 })),
    ];

    println!("{:<14} {:>10} {:>12} {:>12}",
             "strategy", "build_s", "replicas", "post_MiB");
    let mut indexes: Vec<(String, IvfIndex, f64)> = Vec::new();
    for (name, a) in &variants {
        let t0 = Instant::now();
        let idx = IvfIndex::build(data.clone(), centroids.clone(), a.as_ref());
        let build_s = t0.elapsed().as_secs_f32();
        let post_mib = idx.posting_bytes() as f64 / (1024.0 * 1024.0);
        println!("{:<14} {:>10.2} {:>12} {:>12.3}",
                 name, build_s, idx.replicas, post_mib);
        indexes.push((name.to_string(), idx, build_s as f64));
    }
    println!();

    // --- Search sweep ---
    println!("{:<14} {:>6} {:>10} {:>12}",
             "strategy", "nprobe", "recall@10", "us/query");
    for (name, idx, _bs) in &indexes {
        let rows = eval(idx, &queries, &truths, k, &nprobes);
        for (np, r, us) in rows {
            println!("{:<14} {:>6} {:>10.4} {:>12.2}", name, np, r, us);
        }
    }

    println!();
    println!("# recall table (rows = strategy, cols = nprobe)");
    print!("{:<14}", "strategy");
    for &np in &nprobes { print!("{:>10}", np); }
    println!();
    for (name, idx, _bs) in &indexes {
        print!("{:<14}", name);
        let rows = eval(idx, &queries, &truths, k, &nprobes);
        for (_, r, _) in rows { print!("{:>10.4}", r); }
        println!();
    }
}
