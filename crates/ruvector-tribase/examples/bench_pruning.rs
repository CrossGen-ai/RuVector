//! Benchmark: Flat vs IVF vs Tribase on synthetic data.
//!
//! Reports wall time, distance computations, and recall@k vs Flat ground truth.
//! Run with: cargo run --release -p ruvector-tribase --example bench_pruning

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ruvector_tribase::{FlatIndex, IvfIndex, TribaseIndex};
use std::time::Instant;

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Mixture-of-Gaussians: realistic structure where triangle bounds work.
    // 50 clusters spread on [-5, 5]^dim, sigma=0.4 isotropic noise.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let nc = 50usize;
    let centers: Vec<Vec<f32>> = (0..nc)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % nc];
            c.iter()
                .map(|&v| v + (rng.gen::<f32>() * 2.0 - 1.0) * 0.4)
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Run {
    method: &'static str,
    qps: f64,
    avg_dists: f64,
    recall: f64,
    p99_us: f64,
}

fn recall_at_k(gt: &[(f32, u32)], got: &[(f32, u32)]) -> f64 {
    use std::collections::HashSet;
    let g: HashSet<u32> = gt.iter().map(|(_, i)| *i).collect();
    let hits = got.iter().filter(|(_, i)| g.contains(i)).count();
    hits as f64 / gt.len() as f64
}

fn run_bench(n: usize, dim: usize, n_lists: usize, nprobe: usize, k: usize, n_queries: usize) {
    println!(
        "\n=== n={} dim={} n_lists={} nprobe={} k={} q={} ===",
        n, dim, n_lists, nprobe, k, n_queries
    );

    let data = synth(n, dim, 42);
    let queries = synth(n_queries, dim, 99);

    // --- Build ---
    let t0 = Instant::now();
    let flat = FlatIndex::build(data.clone());
    let t_flat_build = t0.elapsed();
    let t0 = Instant::now();
    let ivf = IvfIndex::build(data.clone(), n_lists, 10, 42);
    let t_ivf_build = t0.elapsed();
    let t0 = Instant::now();
    let tri = TribaseIndex::build(data.clone(), n_lists, 10, 42);
    let t_tri_build = t0.elapsed();

    println!(
        "build: flat={:?} ivf={:?} tribase={:?}",
        t_flat_build, t_ivf_build, t_tri_build
    );

    // Tribase memory overhead: 4 bytes per vector (f32 residual)
    let tribase_overhead = n * std::mem::size_of::<f32>();
    let raw_data_bytes = n * dim * std::mem::size_of::<f32>();
    println!(
        "memory: raw={:.2} MiB, tribase overhead={:.2} KiB ({:.3}% of raw)",
        raw_data_bytes as f64 / (1024.0 * 1024.0),
        tribase_overhead as f64 / 1024.0,
        100.0 * tribase_overhead as f64 / raw_data_bytes as f64
    );

    // --- Ground truth (flat) ---
    let mut gt: Vec<Vec<(f32, u32)>> = Vec::with_capacity(n_queries);
    let t0 = Instant::now();
    let mut flat_dists_total: u64 = 0;
    let mut flat_latencies = Vec::with_capacity(n_queries);
    for q in &queries {
        let qt = Instant::now();
        let (r, s) = flat.search(q, k);
        flat_latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        flat_dists_total += s.dist_computations;
        gt.push(r);
    }
    let t_flat = t0.elapsed();

    // --- IVF ---
    let t0 = Instant::now();
    let mut ivf_dists_total: u64 = 0;
    let mut ivf_recall_sum = 0.0;
    let mut ivf_latencies = Vec::with_capacity(n_queries);
    for (i, q) in queries.iter().enumerate() {
        let qt = Instant::now();
        let (r, s) = ivf.search(q, k, nprobe);
        ivf_latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        ivf_dists_total += s.dist_computations;
        ivf_recall_sum += recall_at_k(&gt[i], &r);
    }
    let t_ivf = t0.elapsed();

    // --- Tribase ---
    let t0 = Instant::now();
    let mut tri_dists_total: u64 = 0;
    let mut tri_recall_sum = 0.0;
    let mut tri_latencies = Vec::with_capacity(n_queries);
    for (i, q) in queries.iter().enumerate() {
        let qt = Instant::now();
        let (r, s) = tri.search(q, k, nprobe);
        tri_latencies.push(qt.elapsed().as_secs_f64() * 1e6);
        tri_dists_total += s.dist_computations;
        tri_recall_sum += recall_at_k(&gt[i], &r);
    }
    let t_tri = t0.elapsed();

    let p99 = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64) * 0.99) as usize]
    };

    let runs = [
        Run {
            method: "flat",
            qps: n_queries as f64 / t_flat.as_secs_f64(),
            avg_dists: flat_dists_total as f64 / n_queries as f64,
            recall: 1.0,
            p99_us: p99(flat_latencies),
        },
        Run {
            method: "ivf",
            qps: n_queries as f64 / t_ivf.as_secs_f64(),
            avg_dists: ivf_dists_total as f64 / n_queries as f64,
            recall: ivf_recall_sum / n_queries as f64,
            p99_us: p99(ivf_latencies),
        },
        Run {
            method: "tribase",
            qps: n_queries as f64 / t_tri.as_secs_f64(),
            avg_dists: tri_dists_total as f64 / n_queries as f64,
            recall: tri_recall_sum / n_queries as f64,
            p99_us: p99(tri_latencies),
        },
    ];

    println!(
        "{:<10} {:>10} {:>14} {:>10} {:>12}",
        "method", "QPS", "avg dists", "recall", "p99 (us)"
    );
    for r in &runs {
        println!(
            "{:<10} {:>10.1} {:>14.0} {:>10.3} {:>12.1}",
            r.method, r.qps, r.avg_dists, r.recall, r.p99_us
        );
    }
    let ivf_run = runs[1];
    let tri_run = runs[2];
    let dist_reduction = 1.0 - tri_run.avg_dists / ivf_run.avg_dists;
    let speedup = tri_run.qps / ivf_run.qps;
    println!(
        "tribase vs ivf: {:.1}% fewer dists, {:.2}x QPS speedup, recall delta {:.4}",
        dist_reduction * 100.0,
        speedup,
        tri_run.recall - ivf_run.recall
    );
}

fn main() {
    println!("ruvector-tribase benchmark");
    println!("hardware: {}", std::env::var("HW").unwrap_or_else(|_| "unspecified (set HW env var)".into()));

    // Sweep across realistic configurations
    run_bench(10_000, 64, 64, 8, 10, 200);
    run_bench(20_000, 128, 128, 16, 10, 200);
    run_bench(50_000, 128, 256, 24, 10, 100);
    run_bench(20_000, 128, 128, 32, 10, 200);
}
