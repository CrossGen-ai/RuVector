//! Three-variant benchmark: float32 vs global-sq4 vs per-cluster-sq4.
//!
//! Measures index size (code bytes + table overhead), recall@10 vs exact,
//! per-query latency, and mean squared reconstruction error of residuals.

use per_cluster_sq_ivf::data::{make_gaussian_mixture, exact_topk};
use per_cluster_sq_ivf::ivf::{
    Float32Backend, GlobalSq4Backend, IvfBackend, IvfIndex, PerClusterSq4Backend,
};
use std::time::Instant;

fn recall(gt: &[u32], got: &[u32]) -> f32 {
    let mut hits = 0;
    for g in got {
        if gt.contains(g) {
            hits += 1;
        }
    }
    hits as f32 / gt.len() as f32
}

#[derive(Debug)]
struct Stats {
    name: &'static str,
    code_bytes_per_vec: f64,
    overhead_bytes_total: f64,
    total_bytes: f64,
    recall10: f32,
    mean_recon_err: f64,
    mean_qps: f64,
    mean_latency_us: f64,
}

fn run<B: IvfBackend>(
    label: &'static str,
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    queries: &[f32],
    gt: &[Vec<u32>],
    nprobe: usize,
    backend: B,
    overhead_bytes_total: f64,
) -> Stats {
    let t0 = Instant::now();
    let idx = IvfIndex::train(data, n, d, k, 25, 1, backend);
    let train_ms = t0.elapsed().as_millis();
    eprintln!("[{}] train: {} ms", label, train_ms);

    let code_bytes_per_vec = idx.backend.storage_bytes_per_vec(d);

    // Reconstruction error on 512 points
    let m = 512.min(n);
    let mut err_sum = 0.0f64;
    for i in 0..m {
        let v = &data[i * d..(i + 1) * d];
        // nearest centroid
        let mut c_best = 0usize;
        let mut cd = f32::INFINITY;
        for c in 0..k {
            let dist = per_cluster_sq_ivf::kmeans::l2_sq(v, &idx.centroids[c * d..(c + 1) * d]);
            if dist < cd {
                cd = dist;
                c_best = c;
            }
        }
        let mut res = vec![0.0f32; d];
        for j in 0..d {
            res[j] = v[j] - idx.centroids[c_best * d + j];
        }
        // local index
        let local = idx.cluster_ids[c_best]
            .iter()
            .position(|&g| g as usize == i)
            .expect("self must be in its cluster");
        // self-score equals residual reconstruction squared error vs zero-residual query
        let zero_q = vec![0.0f32; d];
        let s_quant = idx.backend.score(c_best as u32, local, &zero_q);
        let s_exact = per_cluster_sq_ivf::kmeans::l2_sq(&res, &zero_q);
        err_sum += ((s_quant - s_exact).abs()) as f64;
    }
    let mean_recon_err = err_sum / m as f64;

    // Recall + latency
    let nq = gt.len();
    let mut recall_sum = 0.0f32;
    let t1 = Instant::now();
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let got = idx.search(q, 10, nprobe);
        let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        recall_sum += recall(&gt[qi], &got_ids);
    }
    let elapsed = t1.elapsed();
    let latency_us = elapsed.as_micros() as f64 / nq as f64;
    let qps = 1e6 / latency_us;

    let total_bytes = code_bytes_per_vec * n as f64 + overhead_bytes_total;

    Stats {
        name: label,
        code_bytes_per_vec,
        overhead_bytes_total,
        total_bytes,
        recall10: recall_sum / nq as f32,
        mean_recon_err,
        mean_qps: qps,
        mean_latency_us: latency_us,
    }
}

fn main() {
    let n = 20_000;
    let d = 64;
    let k = 64;
    let nprobe = 8;
    let nq = 200;
    eprintln!("Generating {} vectors, d={}, k={} clusters, nprobe={}", n, d, k, nprobe);
    let (data, _) = make_gaussian_mixture(n, d, 32, 7);
    let (queries, _) = make_gaussian_mixture(nq, d, 32, 11);

    eprintln!("Computing exact ground truth...");
    let gt: Vec<Vec<u32>> = (0..nq)
        .map(|qi| {
            let q = &queries[qi * d..(qi + 1) * d];
            exact_topk(&data, d, q, 10).into_iter().map(|x| x.0).collect()
        })
        .collect();

    // Variants: compute overhead too.
    let f32_overhead = 0.0;
    let global_overhead = (d * 2 * 4) as f64; // lo, hi per dim as f32
    let per_cluster_overhead = (k * d * 2 * 4) as f64; // per cluster lo, hi

    let mut all = Vec::new();
    all.push(run(
        "float32-baseline",
        &data, n, d, k, &queries, &gt, nprobe,
        Float32Backend::new(), f32_overhead,
    ));
    all.push(run(
        "global-sq4",
        &data, n, d, k, &queries, &gt, nprobe,
        GlobalSq4Backend::new(), global_overhead,
    ));
    all.push(run(
        "per-cluster-sq4",
        &data, n, d, k, &queries, &gt, nprobe,
        PerClusterSq4Backend::new(), per_cluster_overhead,
    ));

    println!("\n=== RESULTS n={} d={} k={} nprobe={} nq={} ===", n, d, k, nprobe, nq);
    println!(
        "{:<22} | {:>10} | {:>13} | {:>13} | {:>8} | {:>14} | {:>10} | {:>12}",
        "backend", "code_B/vec", "overhead_B", "total_B", "recall@10", "mean|Δresid|", "qps", "lat_us/q"
    );
    println!("{}", "-".repeat(130));
    for s in &all {
        println!(
            "{:<22} | {:>10.2} | {:>13.0} | {:>13.0} | {:>8.4} | {:>14.6e} | {:>10.1} | {:>12.2}",
            s.name,
            s.code_bytes_per_vec,
            s.overhead_bytes_total,
            s.total_bytes,
            s.recall10,
            s.mean_recon_err,
            s.mean_qps,
            s.mean_latency_us
        );
    }

    // Numeric acceptance: per-cluster must beat global on recall at the same bit budget.
    let g = all.iter().find(|s| s.name == "global-sq4").unwrap();
    let pc = all.iter().find(|s| s.name == "per-cluster-sq4").unwrap();
    println!("\n--- ACCEPTANCE ---");
    println!(
        "per-cluster recall ({:.4}) vs global recall ({:.4}): Δ = {:+.4}",
        pc.recall10, g.recall10, pc.recall10 - g.recall10
    );
    println!(
        "per-cluster recon err ({:.3e}) vs global ({:.3e}): ratio = {:.3}x",
        pc.mean_recon_err, g.mean_recon_err,
        pc.mean_recon_err / g.mean_recon_err.max(1e-30)
    );
    if pc.recall10 >= g.recall10 && pc.mean_recon_err <= g.mean_recon_err {
        println!("PASS: per-cluster-sq4 >= global-sq4 on BOTH recall and reconstruction.");
    } else if pc.mean_recon_err <= g.mean_recon_err {
        println!(
            "PARTIAL: per-cluster beats global on reconstruction but not recall; see research doc for context."
        );
    } else {
        println!("FAIL: per-cluster did not beat global.");
    }
}
