//! `cargo run --release -p ruvector-nsg --bin nsg-demo`
//!
//! Builds a synthetic clustered dataset, constructs three NSG variants, and
//! reports build time, search QPS, and recall@10 vs brute force.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_nsg::{brute_force_topk, recall_at_k, NsgBuilder, NsgParams};
use std::time::Instant;

fn gauss_clusters(n: usize, d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..c)
        .map(|_| (0..d).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let cc = &centers[rng.gen_range(0..c)];
            cc.iter().map(|x| x + rng.gen_range(-1.5..1.5)).collect()
        })
        .collect()
}

fn uniform(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench(name: &str, params: NsgParams, data: &[Vec<f32>], queries: &[Vec<f32>], gt: &[Vec<ruvector_nsg::SearchHit>], k: usize, l_search: usize) {
    let t0 = Instant::now();
    let index = NsgBuilder::new(params).build(data.to_vec()).expect("build");
    let build_s = t0.elapsed().as_secs_f64();
    let degree = index.avg_degree();
    let graph_kb = index.graph_bytes() as f64 / 1024.0;

    // Warm-up
    for q in queries.iter().take(8) {
        let _ = index.search(q, k, l_search).unwrap();
    }
    let t0 = Instant::now();
    let mut recalls = 0.0f32;
    let mut total_hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let r = index.search(q, k, l_search).unwrap();
        recalls += recall_at_k(&gt[i], &r);
        total_hits += r.len();
    }
    let qps_s = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / qps_s;
    println!(
        "{:>20}  R={:>3} L_build={:>4} L_search={:>4} | build={:>6.2}s  deg={:>5.1}  graph={:>7.1} KiB  recall@{}={:.3}  QPS={:>8.0}  ({} hits)",
        name,
        params.r,
        params.l_build,
        l_search,
        build_s,
        degree,
        graph_kb,
        k,
        recalls / queries.len() as f32,
        qps,
        total_hits
    );
}

fn main() {
    let n: usize = std::env::var("NSG_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let d: usize = std::env::var("NSG_D").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let nq: usize = std::env::var("NSG_QUERIES").ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let k: usize = 10;

    println!(
        "ruvector-nsg demo  ·  N={} D={} queries={} k={}",
        n, d, nq, k
    );
    println!("------------------------------------------------------------");

    let dataset = std::env::var("NSG_DATA").unwrap_or_else(|_| "uniform".into());
    let mut all = match dataset.as_str() {
        "gauss" => gauss_clusters(n + nq, d, 24, 0x511f7),
        _ => uniform(n + nq, d, 0x511f7),
    };
    println!("dataset: {} (NSG_DATA=uniform|gauss to switch)", dataset);
    let queries: Vec<Vec<f32>> = all.split_off(n);
    let data = all;

    // Ground truth (brute force).
    let t0 = Instant::now();
    let gt: Vec<Vec<ruvector_nsg::SearchHit>> =
        queries.iter().map(|q| brute_force_topk(&data, q, k)).collect();
    let bf_s = t0.elapsed().as_secs_f64();
    let bf_qps = nq as f64 / bf_s;
    println!(
        "{:>20}  -                                | build=     -    deg=   -    graph=      -    recall@{}=1.000  QPS={:>8.0}  (baseline)",
        "brute-force", k, bf_qps
    );

    bench(
        "NSG/small",
        NsgParams { r: 20, l_build: 40, k_knn: 30, knn_iters: 6, knn_sample: 1.0, seed: 1, alpha: 1.0 },
        &data, &queries, &gt, k, 40,
    );
    bench(
        "NSG/medium",
        NsgParams { r: 32, l_build: 100, k_knn: 50, knn_iters: 6, knn_sample: 1.0, seed: 1, alpha: 1.2 },
        &data, &queries, &gt, k, 100,
    );
    bench(
        "NSG/large",
        NsgParams { r: 48, l_build: 200, k_knn: 60, knn_iters: 6, knn_sample: 1.0, seed: 1, alpha: 1.4 },
        &data, &queries, &gt, k, 200,
    );

    println!("------------------------------------------------------------");
    println!("Acceptance: NSG/medium recall@{} ≥ 0.90 with QPS > brute-force.", k);
}
