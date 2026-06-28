//! `cargo run --release -p ruvector-lorann --bin bench`
//!
//! Real benchmark binary. No mocks. Prints per-variant build time, mean query
//! latency over `n_queries` queries, and recall@10 against the brute-force
//! ground truth. Numbers from this binary are quoted verbatim in the research
//! document.

use std::time::Instant;

use ruvector_lorann::{
    BruteForceIndex, InnerProductIndex, IvfConfig, IvfIndex, Lcg, LoRannConfig, LoRannIndex,
};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut x = vec![0.0_f32; n * d];
    for k in 0..n * d {
        x[k] = rng.next_f32() - 0.5;
    }
    x
}

/// Clustered dataset — `n` points split across `k_true` ground-truth clusters
/// in a low-rank subspace plus Gaussian-ish noise. This is the regime LoRANN
/// was designed for and the regime real embedding datasets live in.
fn synth_clustered(n: usize, d: usize, k_true: usize, noise: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    // Random unit centers.
    let mut centers = vec![0.0_f32; k_true * d];
    for k in 0..k_true * d {
        centers[k] = rng.next_f32() - 0.5;
    }
    for c in 0..k_true {
        let mut s = 0.0_f32;
        for j in 0..d {
            s += centers[c * d + j] * centers[c * d + j];
        }
        let inv = 1.0 / s.sqrt().max(1e-9);
        for j in 0..d {
            centers[c * d + j] *= inv;
        }
    }
    let mut x = vec![0.0_f32; n * d];
    for i in 0..n {
        let c = i % k_true;
        for j in 0..d {
            x[i * d + j] = centers[c * d + j] + noise * (rng.next_f32() - 0.5);
        }
    }
    x
}

fn recall_at_k(truth: &[Vec<u32>], pred: &[Vec<u32>], k: usize) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (t, p) in truth.iter().zip(pred.iter()) {
        let pset: std::collections::HashSet<u32> = p.iter().copied().take(k).collect();
        for &id in t.iter().take(k) {
            if pset.contains(&id) {
                hits += 1;
            }
            total += 1;
        }
    }
    hits as f32 / total.max(1) as f32
}

fn time<F: FnMut()>(mut f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}

fn bench_search<I: InnerProductIndex>(
    idx: &I,
    queries: &[f32],
    n_queries: usize,
    d: usize,
    k: usize,
) -> (Vec<Vec<u32>>, f64) {
    let mut preds: Vec<Vec<u32>> = Vec::with_capacity(n_queries);
    let elapsed = time(|| {
        preds.clear();
        for i in 0..n_queries {
            let r = idx.search(&queries[i * d..(i + 1) * d], k);
            preds.push(r.into_iter().map(|n| n.id).collect());
        }
    });
    let mean_ms = elapsed * 1000.0 / n_queries as f64;
    (preds, mean_ms)
}

fn run_suite(label: &str, x: &[f32], queries: &[f32], n: usize, d: usize, n_queries: usize, k: usize) {
    println!("\n== {label} ==  n={n}  d={d}  queries={n_queries}  k={k}");

    let mut brute = BruteForceIndex::default();
    let build_brute = time(|| brute.train(x.to_vec(), n, d));
    let (truth, ms_brute) = bench_search(&brute, queries, n_queries, d, k);
    println!(
        "brute        build={:.3}s  query_ms={:.3}  recall@{k}=1.000",
        build_brute, ms_brute
    );

    let ivf_cfg = IvfConfig { n_clusters: 64, nprobe: 8, kmeans_iters: 25, seed: 11 };
    let mut ivf = IvfIndex::new(ivf_cfg);
    let build_ivf = time(|| ivf.train(x.to_vec(), n, d));
    let (ivf_pred, ms_ivf) = bench_search(&ivf, queries, n_queries, d, k);
    let rec_ivf = recall_at_k(&truth, &ivf_pred, k);
    println!(
        "ivf          build={:.3}s  query_ms={:.3}  recall@{k}={:.3}  (clusters={}, nprobe={})",
        build_ivf, ms_ivf, rec_ivf, ivf_cfg.n_clusters, ivf_cfg.nprobe
    );

    for &rank in &[8usize, 16, 32] {
        let cfg = LoRannConfig {
            ivf: ivf_cfg,
            rank,
            subspace_iters: 20,
            rerank_per_cluster: 32,
        };
        let mut lorann = LoRannIndex::new(cfg);
        let build_lr = time(|| lorann.train(x.to_vec(), n, d));
        let (lr_pred, ms_lr) = bench_search(&lorann, queries, n_queries, d, k);
        let rec_lr = recall_at_k(&truth, &lr_pred, k);
        let speedup = ms_brute / ms_lr;
        println!(
            "lorann r={rank:>3}  build={:.3}s  query_ms={:.3}  recall@{k}={:.3}  speedup_vs_brute={:.2}x",
            build_lr, ms_lr, rec_lr, speedup
        );
    }
}

fn main() {
    // Sizes are tuned so build+run completes in seconds on a laptop, while
    // still producing meaningful recall and a measurable QPS gap. The research
    // doc explains why these are honest signals.
    let n = 8_000;
    let d = 128;
    let n_queries = 200;
    let k = 10;

    // Uniform random — adversarial (no cluster structure to exploit).
    let x = synth(n, d, 1);
    let queries = synth(n_queries, d, 2);
    run_suite("UNIFORM (adversarial)", &x, &queries, n, d, n_queries, k);

    // Clustered — the embedding-data regime.
    let x_c = synth_clustered(n, d, 32, 0.15, 1);
    let queries_c = synth_clustered(n_queries, d, 32, 0.15, 2);
    run_suite("CLUSTERED (realistic)", &x_c, &queries_c, n, d, n_queries, k);

    println!("\nAcceptance criteria (CLUSTERED suite):");
    println!("  * lorann r=32 should reach >0.85 recall@{k}");
    println!("  * lorann r=8 should achieve >5x speedup vs brute");
}
