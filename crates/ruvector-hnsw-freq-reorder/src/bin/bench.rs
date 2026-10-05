//! Real benchmark: 3 variants on synthetic clustered vectors, reporting
//! recall@10 and mean per-query latency + distance-computation count.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use ruvector_hnsw_freq_reorder::{
    brute_force_topk, recall, BuildParams, Index, NeighborOrder, SearchParams,
};

/// Synthetic dataset: `nc` isotropic Gaussian clusters in dim `d`.
fn gen_clusters(n: usize, d: usize, nc: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..nc)
        .map(|_| (0..d).map(|_| (rng.gen::<f32>() - 0.5) * 10.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..nc)];
            (0..d).map(|i| c[i] + (rng.gen::<f32>() - 0.5) * 1.0).collect()
        })
        .collect()
}

fn run_variant(
    idx: &Index,
    queries: &[Vec<f32>],
    truths: &[Vec<(f32, u32)>],
    order: NeighborOrder,
    ef: usize,
    k: usize,
) -> (f32, f64, f64) {
    idx.reset_dist_count();
    let sp = SearchParams { ef, k, order };
    let mut total_recall = 0.0f32;
    let start = Instant::now();
    for (q, t) in queries.iter().zip(truths.iter()) {
        let out = idx.search(q, sp);
        total_recall += recall(&out, t);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mean_us = 1e6 * elapsed / queries.len() as f64;
    let mean_dist = idx.dist_count.get() as f64 / queries.len() as f64;
    (total_recall / queries.len() as f32, mean_us, mean_dist)
}

fn main() {
    let n = 20_000usize;
    let d = 64usize;
    let nc = 64usize;
    let m = 24usize;
    let seed = 0xBEEF_CAFE_u64;

    println!("=== ruvector-hnsw-freq-reorder — synthetic benchmark ===");
    println!("dataset : n={n} dim={d} clusters={nc}");
    println!("graph   : M={m} ef_build=96");

    let vecs = gen_clusters(n, d, nc, seed);
    let n_train = 1_000usize;
    let n_query = 500usize;
    let train: Vec<Vec<f32>> = vecs[..n_train].to_vec();
    let queries: Vec<Vec<f32>> = {
        // Fresh queries drawn from the same distribution but not in the
        // training set so warmup doesn't overfit.
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_A5A5);
        (0..n_query)
            .map(|_| {
                let c = &vecs[rng.gen_range(0..vecs.len())];
                (0..d).map(|i| c[i] + (rng.gen::<f32>() - 0.5) * 0.5).collect()
            })
            .collect()
    };

    // Build once.
    let t0 = Instant::now();
    let mut idx = Index::new(d, BuildParams { m, ef_build: 96, seed });
    idx.build(&vecs);
    println!("build   : {:.2}s", t0.elapsed().as_secs_f64());

    // Ground truth.
    let t1 = Instant::now();
    let truths: Vec<Vec<(f32, u32)>> = queries.iter().map(|q| brute_force_topk(&idx.store, q, 10)).collect();
    println!("truth   : {:.2}s (brute force top-10)", t1.elapsed().as_secs_f64());

    // Warmup + reorder.
    let t2 = Instant::now();
    idx.warmup(&train, 32);
    idx.reorder_by_frequency();
    println!("warmup  : {:.2}s ({} training queries)", t2.elapsed().as_secs_f64(), n_train);

    // Report freq-distribution stats to make the effect visible.
    let (mut nonzero, mut total) = (0u64, 0u64);
    let mut top_share = 0.0f64;
    for (i, row) in idx.freq.iter().enumerate() {
        let sum: u64 = row.iter().map(|&x| x as u64).sum();
        if sum > 0 {
            nonzero += 1;
            total += sum;
            let head = idx.adj[i].len().min(4);
            let head_sum: u64 = row[..head].iter().map(|&x| x as u64).sum();
            top_share += head_sum as f64 / sum as f64;
        }
    }
    let avg_top_share = if nonzero > 0 { top_share / nonzero as f64 } else { 0.0 };
    println!(
        "reorder : {nonzero} nodes with hits, {total} total edge hits, top-4/all share={:.2}",
        avg_top_share
    );

    println!("\n{:<32} {:>10} {:>12} {:>12}", "variant/ef", "recall@10", "latency_us", "dist/query");
    for &ef in &[16usize, 32, 64] {
        for &(name, order) in &[
            ("Natural",              NeighborOrder::Natural),
            ("FreqReorder",          NeighborOrder::FreqReorder),
            ("FreqReorderPrefetch",  NeighborOrder::FreqReorderPrefetch),
        ] {
            let (r, us, dq) = run_variant(&idx, &queries, &truths, order, ef, 10);
            println!("{:<24} ef={:<3} {:>10.3} {:>12.2} {:>12.1}", name, ef, r, us, dq);
        }
        println!();
    }
}
