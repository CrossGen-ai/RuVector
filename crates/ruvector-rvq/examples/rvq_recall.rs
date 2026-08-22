//! Recall-vs-compression sweep for RVQ: prints a small table you can drop
//! straight into a research doc.
//!
//! Run: `cargo run --release -p ruvector-rvq --example rvq_recall`

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_rvq::{Rvq, RvqConfig, RvqIndex};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// Mixture-of-Gaussians: closer to real embedding distributions (which are
/// clustered around a few semantic modes rather than i.i.d. uniform).
fn synth_clustered(n: usize, d: usize, n_modes: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Sample n_modes mode centres uniformly in [-1, 1]^d.
    let modes: Vec<f32> = (0..n_modes * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n {
        let m = rng.gen_range(0..n_modes);
        for j in 0..d {
            let mu = modes[m * d + j];
            // Box-Muller-ish crude gaussian
            let u1: f32 = rng.gen_range(1e-6f32..1.0);
            let u2: f32 = rng.gen_range(0.0f32..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            out.push(mu + sigma * z);
        }
    }
    out
}

fn brute_l2_topk(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / d;
    let mut scored: Vec<(u32, f32)> =
        (0..n).map(|i| {
            let mut s = 0f32;
            for j in 0..d { let e = data[i * d + j] - q[j]; s += e * e; }
            (i as u32, s)
        }).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn main() {
    let d = 128;
    let n = 8_000;
    let n_q = 100;
    let k = 10;
    // Use clustered data (n_modes=32 semantic clusters, sigma=0.15) to
    // approximate real embedding distributions; i.i.d. uniform is
    // worst-case for any quantizer since distances all collapse.
    let data = synth_clustered(n, d, 32, 0.15, 3);
    let q = synth_clustered(n_q, d, 32, 0.15, 4);
    let _ = synth; // silence unused-import lint if we drop this later

    // Precompute truth
    let truth: Vec<std::collections::HashSet<u32>> =
        (0..n_q).map(|i| brute_l2_topk(&data, d, &q[i * d..(i + 1) * d], k).into_iter().collect()).collect();

    println!("=== Pure RVQ (no reranking) ===");
    println!("stages  bytes/vec  ratio    recall@10   mse");
    println!("------  ---------  -------  ----------  ------");
    for &stages in &[2usize, 4, 6, 8, 12, 16] {
        let cfg = RvqConfig { stages, k: 256, kmeans_iters: 12, seed: 42 };
        let rvq = Rvq::train(&data, n, d, &cfg).unwrap();
        let mse = rvq.reconstruction_mse(&data, n);
        let idx = RvqIndex::build(rvq, &data);
        let mut hits = 0usize;
        for qi in 0..n_q {
            let got = idx.search_l2(&q[qi * d..(qi + 1) * d], k);
            for r in &got { if truth[qi].contains(&r.id) { hits += 1; } }
        }
        let recall = hits as f32 / (n_q * k) as f32;
        let ratio = (d * 4) as f32 / stages as f32;
        println!("{stages:>6}  {stages:>9}  {ratio:>5.1}x  {recall:>10.3}  {mse:>6.4}");
    }

    println!("\n=== RVQ + rerank top-N with full precision (production recipe) ===");
    println!("stages  rerank_N   recall@10");
    println!("------  --------   ---------");
    for &stages in &[4usize, 8, 16] {
        let cfg = RvqConfig { stages, k: 256, kmeans_iters: 12, seed: 42 };
        let rvq = Rvq::train(&data, n, d, &cfg).unwrap();
        let idx = RvqIndex::build(rvq, &data);
        for &rerank in &[50usize, 100, 200, 500] {
            let mut hits = 0usize;
            for qi in 0..n_q {
                let got = idx.search_l2_rerank(&q[qi * d..(qi + 1) * d], k, rerank, &data);
                for r in &got { if truth[qi].contains(&r.id) { hits += 1; } }
            }
            let recall = hits as f32 / (n_q * k) as f32;
            println!("{stages:>6}  {rerank:>8}   {recall:>9.3}");
        }
    }
}
