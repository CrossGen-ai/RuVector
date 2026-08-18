//! End-to-end benchmark: build once, reorder four ways, measure search.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_hnsw_reorder::reorder::log_gap_cost;
use ruvector_hnsw_reorder::{
    apply_permutation, bfs_order, build_hnsw, gorder, identity_order, rgb_order, search_knn,
};
use std::time::Instant;

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

fn recall(pred: &[u32], truth: &[u32], k: usize) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().take(k).copied().collect();
    let hits = pred.iter().take(k).filter(|x| t.contains(x)).count();
    hits as f32 / k as f32
}

fn brute_force(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut d: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let s = i * dim;
            let mut ss = 0f32;
            for j in 0..dim {
                let x = data[s + j] - q[j];
                ss += x * x;
            }
            (ss, i as u32)
        })
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(k).map(|x| x.1).collect()
}

fn main() {
    // Tunables via env to keep default runtime bounded.
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
    let m: usize = 24;
    let ef_c: usize = 96;
    let ef_s: usize = 64;
    let k: usize = 10;
    let n_queries: usize = 500;

    println!("== ruvector-hnsw-reorder bench ==");
    println!("n={n} dim={dim} m={m} ef_c={ef_c} ef_s={ef_s} k={k} queries={n_queries}");

    let data = gen_data(n, dim, 42);
    let queries = gen_data(n_queries, dim, 43);

    let t0 = Instant::now();
    let g_natural = build_hnsw(data.clone(), dim, m, ef_c);
    println!("build: {:.2}s   graph_bytes={} MiB", t0.elapsed().as_secs_f64(), g_natural.bytes() / (1024 * 1024));

    // Adversarially shuffle node ids to simulate bulk-loaded graphs where
    // insertion order does NOT encode locality — the realistic hard case.
    let mut rng_perm = StdRng::seed_from_u64(0xC0FFEE);
    let mut shuffle: Vec<u32> = (0..n as u32).collect();
    use rand::seq::SliceRandom;
    shuffle.shuffle(&mut rng_perm);
    let g_shuffled = ruvector_hnsw_reorder::apply_permutation(&g_natural, &shuffle);
    println!("shuffled graph log_gap = {:.3}", ruvector_hnsw_reorder::reorder::log_gap_cost(&g_shuffled));
    let g = g_shuffled;

    // Ground truth via brute force on the graph's actual vector store
    // (post-shuffle) so recall is comparable across all reorderings.
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| brute_force(&g.data, dim, &queries[i * dim..(i + 1) * dim], k))
        .collect();
    println!("ground truth: {:.2}s", t0.elapsed().as_secs_f64());

    let strategies: Vec<(&str, Box<dyn Fn(&_) -> Vec<u32>>)> = vec![
        ("identity", Box::new(|g: &_| identity_order(g))),
        ("bfs", Box::new(|g: &_| bfs_order(g))),
        ("gorder(w=8)", Box::new(|g: &_| gorder(g, 8))),
        ("rgb(d=14)", Box::new(|g: &_| rgb_order(g, 14))),
    ];

    println!(
        "\n{:<14} {:>10} {:>12} {:>10} {:>10} {:>12} {:>10}",
        "strategy", "reorder_ms", "log_gap", "qps", "µs/query", "stride_sum", "recall@10"
    );
    for (name, f) in strategies {
        let t = Instant::now();
        let perm = f(&g);
        let reorder_ms = t.elapsed().as_secs_f64() * 1e3;
        let g2 = apply_permutation(&g, &perm);
        let cost = log_gap_cost(&g2);
        // Map new-id -> old-id for recall bookkeeping.
        let map_back = |ids: &[u32]| -> Vec<u32> { ids.iter().map(|&i| perm[i as usize]).collect() };

        // Warm up.
        for i in 0..64 {
            let _ = search_knn(&g2, &queries[(i % n_queries) * dim..((i % n_queries) + 1) * dim], k, ef_s);
        }
        // Timed loop.
        let iters: usize = 3;
        let mut best_qps = 0f64;
        let mut best_stride = 0u64;
        let mut best_recall = 0f32;
        for _ in 0..iters {
            let t = Instant::now();
            let mut stride_sum = 0u64;
            let mut recall_sum = 0f32;
            for i in 0..n_queries {
                let (ids, s) = search_knn(&g2, &queries[i * dim..(i + 1) * dim], k, ef_s);
                stride_sum += s.id_stride_sum;
                recall_sum += recall(&map_back(&ids), &truth[i], k);
            }
            let secs = t.elapsed().as_secs_f64();
            let qps = n_queries as f64 / secs;
            if qps > best_qps {
                best_qps = qps;
                best_stride = stride_sum;
                best_recall = recall_sum / n_queries as f32;
            }
        }
        let us = 1e6 / best_qps;
        println!(
            "{:<14} {:>10.1} {:>12.3} {:>10.0} {:>10.2} {:>12} {:>10.4}",
            name, reorder_ms, cost, best_qps, us, best_stride, best_recall
        );
    }
}
