//! Head-to-head benchmark: Sequential vs BlockMax(heuristic) vs BlockMax(tight)
//! over a shared IVF-PQ index.
//!
//! Numbers printed here are the ones cited in the research doc.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use ruvector_blockmax_ivf::{
    BlockMaxBitmaskScanner, BlockMaxScanner, BlockMaxTightScanner, Hit, IvfPqBuilder, PqParams,
    ScanStats, Scanner, SequentialScanner, l2_sq,
};
use std::collections::HashSet;
use std::time::Instant;

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Clustered data: `nclusters` gaussian blobs. More realistic distribution and
/// gives block-max a chance to prune, because points that land in the same
/// coarse cell + same block tend to share subspace code patterns.
fn gen_clustered(n: usize, dim: usize, nclusters: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut centers = vec![0f32; nclusters * dim];
    for v in centers.iter_mut() { *v = rng.gen::<f32>() * 4.0 - 2.0; }
    let mut data = vec![0f32; n * dim];
    for i in 0..n {
        let c = rng.gen_range(0..nclusters);
        for d in 0..dim {
            let noise = (rng.gen::<f32>() - 0.5) * 2.0 * spread;
            data[i * dim + d] = centers[c * dim + d] + noise;
        }
    }
    data
}

fn ground_truth(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> HashSet<u32> {
    let mut v: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| (i, l2_sq(q, &data[i as usize * dim..(i as usize + 1) * dim])))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall(hits: &[Hit], gt: &HashSet<u32>) -> f32 {
    let mut c = 0;
    for h in hits { if gt.contains(&h.id) { c += 1; } }
    c as f32 / gt.len() as f32
}

fn run_bench(dim: usize, n: usize, nq: usize, nlist: usize, nprobe: usize, block_size: usize, m: usize, k: usize, label: &str, clustered: bool) {
    println!("\n=== {label} :: n={n} dim={dim} nlist={nlist} nprobe={nprobe} m={m} block={block_size} k={k} clustered={clustered} ===");
    let (data, queries) = if clustered {
        let d = gen_clustered(n, dim, 32, 0.4, 11);
        let q = gen_clustered(nq, dim, 32, 0.4, 22);
        (d, q)
    } else {
        (gen_data(n, dim, 11), gen_data(nq, dim, 22))
    };

    let t_build = Instant::now();
    let index = IvfPqBuilder {
        nlist, kmeans_iter: 15, block_size,
        pq: PqParams { m, ksub: 256, kmeans_iter: 15 },
        seed: 7,
    }.train(&data, dim);
    println!("build: {:.2}s", t_build.elapsed().as_secs_f64());

    // precompute GT once
    let gts: Vec<HashSet<u32>> = (0..nq)
        .map(|qi| ground_truth(&data, n, dim, &queries[qi * dim..(qi + 1) * dim], k))
        .collect();

    let scanners: Vec<(&str, Box<dyn Scanner>)> = vec![
        ("sequential", Box::new(SequentialScanner)),
        ("blockmax-heur", Box::new(BlockMaxScanner)),
        ("blockmax-tight", Box::new(BlockMaxTightScanner)),
        ("blockmax-bitmask", Box::new(BlockMaxBitmaskScanner)),
    ];
    for (name, scanner) in scanners {
        let mut total_recall = 0f32;
        let mut agg = ScanStats::default();
        // warm-up
        for qi in 0..nq.min(5) {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let _ = scanner.search(&index, q, k, nprobe);
        }
        let t0 = Instant::now();
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let (hits, s) = scanner.search(&index, q, k, nprobe);
            total_recall += recall(&hits, &gts[qi]);
            agg.blocks_total += s.blocks_total;
            agg.blocks_skipped += s.blocks_skipped;
            agg.codes_evaluated += s.codes_evaluated;
        }
        let dt = t0.elapsed().as_secs_f64();
        let qps = nq as f64 / dt;
        let r = total_recall / nq as f32;
        let skip_pct = 100.0 * agg.blocks_skipped as f64 / agg.blocks_total.max(1) as f64;
        let codes_per_q = agg.codes_evaluated as f64 / nq as f64;
        println!(
            "  {name:<16} time={dt:.3}s qps={qps:>8.1} r@{k}={r:.3} codes/q={codes_per_q:>8.0} skipped={skip_pct:>5.1}%"
        );
    }
}

fn main() {
    // Shape 1: small random.
    run_bench(64, 10_000, 200, 64, 8, 64, 8, 10, "shape-small-random", false);
    // Shape 2: medium random — headline for uniform data (where IVF-PQ struggles anyway).
    run_bench(128, 50_000, 300, 128, 16, 128, 16, 10, "shape-medium-random", false);
    // Shape 3: medium CLUSTERED — realistic distribution where block-max should prune.
    run_bench(128, 50_000, 300, 128, 16, 128, 16, 10, "shape-medium-clustered", true);
    // Shape 4: clustered + higher nprobe (sweep point: more probing → more pruning headroom).
    run_bench(128, 50_000, 300, 128, 32, 128, 16, 10, "shape-clustered-nprobe32", true);
}
