//! Quick smoke demo. Run: cargo run --release -p ruvector-ivf-radial-head-tail --example rht_demo
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use ruvector_ivf_radial_head_tail::{BuildParams, IvfRht, ScanMode, exact_topk, recall_at_k};

fn main() {
    let dim = 32;
    let n = 2_000;
    let mut rng = Pcg64Mcg::new(7);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let idx = IvfRht::build(&data, BuildParams { n_cells: 32, head_fraction: 0.5, kmeans_iters: 8, seed: 11 });
    let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let gt = exact_topk(&data, &q, 10);
    let full = idx.search(&q, 10, 8, ScanMode::Full);
    let adaptive = idx.search(&q, 10, 8, ScanMode::Adaptive);
    println!("full recall@10={:.3}", recall_at_k(&full, &gt));
    println!("adaptive recall@10={:.3}", recall_at_k(&adaptive, &gt));
    println!("memory bytes: {}", idx.memory_bytes());
}
