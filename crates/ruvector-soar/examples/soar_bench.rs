//! soar-bench: real recall@k / QPS sweep across nprobe for 3 assignment modes.
//!
//! Usage: `cargo run --release --example soar-bench -p ruvector-soar`
//!
//! Emits a markdown table on stdout suitable for embedding in the research doc.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_soar::{
    distance::l2_sq, kmeans_lloyd, Assignment, IvfIndex, KMeansConfig, SoarConfig,
};
use std::time::Instant;

fn gaussian_mixture(n: usize, d: usize, n_modes: usize, seed: u64) -> Vec<f32> {
    // Cluster centers in a hypercube; samples in unit-radius Gaussian-ish balls.
    let mut r = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_modes)
        .map(|_| (0..d).map(|_| r.gen::<f32>() * 8.0 - 4.0).collect())
        .collect();
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n {
        let c = &centers[r.gen_range(0..n_modes)];
        for j in 0..d {
            // Box-Muller-ish: sum of 3 uniforms approximates Gaussian.
            let z = (r.gen::<f32>() + r.gen::<f32>() + r.gen::<f32>() - 1.5) * 2.0;
            out.push(c[j] + z);
        }
    }
    out
}

fn brute_top_k(data: &[f32], n: usize, d: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut s: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, l2_sq(&data[i * d..(i + 1) * d], q)))
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall(got: &[u32], truth: &[u32]) -> f32 {
    let mut hit = 0;
    for t in truth {
        if got.contains(t) { hit += 1; }
    }
    hit as f32 / truth.len() as f32
}

fn run_variant(
    name: &str,
    centroids: &[f32],
    n_lists: usize,
    dim: usize,
    data: &[f32],
    queries: &[f32],
    truth: &[Vec<u32>],
    k: usize,
    nprobes: &[usize],
    assignment: Assignment,
) {
    let t_build = Instant::now();
    let idx = IvfIndex::build(
        centroids.to_vec(),
        n_lists,
        dim,
        data.to_vec(),
        &SoarConfig { assignment },
    )
    .unwrap();
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    let overhead_mb = idx.index_overhead_bytes() as f64 / (1024.0 * 1024.0);
    let total_mb = idx.total_bytes() as f64 / (1024.0 * 1024.0);

    let nq = queries.len() / dim;
    println!(
        "\n### {} — build {:.1} ms, index overhead {:.3} MB, total {:.3} MB",
        name, build_ms, overhead_mb, total_mb
    );
    println!("| nprobe | Recall@{} | QPS  | Latency (µs/query) |", k);
    println!("|-------:|----------:|-----:|-------------------:|");

    for &np in nprobes {
        // Warm-up
        for qi in 0..nq.min(8) {
            let _ = idx.search(&queries[qi * dim..(qi + 1) * dim], k, np);
        }
        let t = Instant::now();
        let mut tot_recall = 0.0;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let res: Vec<u32> = idx.search(q, k, np).iter().map(|r| r.id).collect();
            tot_recall += recall(&res, &truth[qi]);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let avg_recall = tot_recall / nq as f32;
        let qps = nq as f64 / elapsed;
        let lat_us = elapsed * 1e6 / nq as f64;
        println!(
            "| {:>6} | {:>9.4} | {:>4.0} | {:>18.1} |",
            np, avg_recall, qps, lat_us
        );
    }
}

fn main() {
    // Dataset shape. Modest sizes keep runtime under ~30s while still showing
    // a clean recall delta.
    let n: usize = 20_000;
    let dim: usize = 64;
    let n_modes = 24;
    let nq: usize = 200;
    let k: usize = 10;
    let n_lists: usize = 64;
    let nprobes: Vec<usize> = vec![1, 2, 4, 8, 16, 32];

    println!("# SOAR — IVF spillover benchmark");
    println!(
        "n={}, dim={}, modes={}, nq={}, k={}, n_lists={}, rustc release",
        n, dim, n_modes, nq, k, n_lists
    );

    let t = Instant::now();
    let data = gaussian_mixture(n, dim, n_modes, 11);
    let queries = gaussian_mixture(nq, dim, n_modes, 22);
    println!("Dataset built in {:.2} s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let truth: Vec<Vec<u32>> = (0..nq)
        .map(|qi| brute_top_k(&data, n, dim, &queries[qi * dim..(qi + 1) * dim], k))
        .collect();
    println!("Brute-force truth in {:.2} s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let centroids = kmeans_lloyd(
        &data,
        n,
        dim,
        &KMeansConfig { k: n_lists, iters: 15, seed: 33 },
    );
    println!("k-means trained in {:.2} s", t.elapsed().as_secs_f64());

    run_variant("Naive IVF (1 cell)", &centroids, n_lists, dim, &data, &queries, &truth, k, &nprobes, Assignment::Naive);
    run_variant("Isotropic spillover (λ=0)", &centroids, n_lists, dim, &data, &queries, &truth, k, &nprobes, Assignment::IsotropicSpillover);
    run_variant("SOAR anisotropic (λ=1.0)", &centroids, n_lists, dim, &data, &queries, &truth, k, &nprobes, Assignment::SoarAnisotropic { lambda: 1.0 });
    run_variant("SOAR anisotropic (λ=4.0)", &centroids, n_lists, dim, &data, &queries, &truth, k, &nprobes, Assignment::SoarAnisotropic { lambda: 4.0 });
}
