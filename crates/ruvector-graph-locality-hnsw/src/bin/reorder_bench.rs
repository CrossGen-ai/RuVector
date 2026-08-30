//! Real (non-mocked) A/B benchmark: identity vs BFS vs RCM reorder.
//!
//! Builds one HNSW index over a synthetic clustered dataset, then measures
//! wall-clock QPS and recall for three physically-permuted copies of that
//! same index. Every knob (n, dim, seeds) is fixed so runs are reproducible.

use ruvector_graph_locality_hnsw::hnsw::{HnswConfig, HnswIndex};
use ruvector_graph_locality_hnsw::order::{
    mean_edge_gap, BfsOrder, IdentityOrder, RcmOrder, Reordered, ReorderStrategy,
};
use ruvector_graph_locality_hnsw::query::{search, SearchStats};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

fn clustered_dataset(n: usize, d: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();
    let normal = Normal::new(0.0, 0.5).unwrap();
    (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            (0..d).map(|j| c[j] + normal.sample(&mut rng) as f32).collect()
        })
        .collect()
}

fn brute_topk(vs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut all: Vec<(f32, u32)> = vs
        .iter()
        .enumerate()
        .map(|(i, v)| (ruvector_graph_locality_hnsw::sq_l2(v, q), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all.into_iter().take(k).map(|(_, i)| i).collect()
}

fn run<S: ReorderStrategy>(
    src: &HnswIndex,
    strat: &S,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
    ef: usize,
) {
    let t0 = Instant::now();
    let re = Reordered::build(src, strat);
    let reorder_ms = t0.elapsed().as_secs_f64() * 1e3;
    let gap = mean_edge_gap(&re.index);
    // Warm up
    let mut warm = SearchStats::default();
    for q in queries.iter().take(50) {
        let _ = search(&re.index, q, k, ef, &mut warm);
    }
    // Measured pass
    let mut stats = SearchStats::default();
    let mut hits = 0usize;
    let n_reps = 3;
    let t1 = Instant::now();
    for _rep in 0..n_reps {
        for (qi, q) in queries.iter().enumerate() {
            let r = search(&re.index, q, k, ef, &mut stats);
            let ids_new: Vec<u32> = r.iter().map(|&(_, id)| id).collect();
            let ids_old: std::collections::HashSet<u32> = ids_new
                .iter()
                .map(|&i| re.new_to_old[i as usize])
                .collect();
            let t: &std::collections::HashSet<u32> =
                &truths[qi].iter().copied().collect();
            hits += ids_old.iter().filter(|x| t.contains(x)).count();
        }
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let total_q = (queries.len() * n_reps) as f64;
    let qps = total_q / elapsed;
    let recall = hits as f64 / (total_q * k as f64);
    let avg_visited = stats.nodes_visited as f64 / total_q;
    println!(
        "  {:<10}  gap={:>9.1}  reorder={:>7.1}ms  QPS={:>9.1}  recall@{}={:.3}  avg_visited={:>6.1}  dist_calls={}",
        strat.name(),
        gap,
        reorder_ms,
        qps,
        k,
        recall,
        avg_visited,
        stats.distance_calls,
    );
}

fn main() {
    let n = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000usize);
    let d = std::env::var("D").ok().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    let n_q = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(500usize);
    let k = 10usize;
    let ef_list: Vec<usize> = std::env::var("EF")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![30, 60, 120]);

    println!("=== graph-locality-hnsw bench (nightly 2026-08-30) ===");
    println!("dataset: n={}, d={}, queries={}, k={}", n, d, n_q, k);

    let t0 = Instant::now();
    let data = clustered_dataset(n, d, 64, 0xBEEF);
    let queries = clustered_dataset(n_q, d, 64, 0xCAFE);
    println!("dataset generation: {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut idx = HnswIndex::new(d, HnswConfig::default());
    for v in &data { idx.insert(v); }
    println!(
        "hnsw build: {:.2}s ({} nodes, entry={:?}, top_layer={})",
        t0.elapsed().as_secs_f64(),
        idx.len(),
        idx.entry_point,
        idx.top_layer,
    );

    let t0 = Instant::now();
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| brute_topk(&data, q, k)).collect();
    println!("brute-force truth: {:.2}s", t0.elapsed().as_secs_f64());

    for &ef in &ef_list {
        println!("\n--- ef={} ---", ef);
        run(&idx, &IdentityOrder, &queries, &truths, k, ef);
        run(&idx, &BfsOrder, &queries, &truths, k, ef);
        run(&idx, &RcmOrder, &queries, &truths, k, ef);
    }
}
