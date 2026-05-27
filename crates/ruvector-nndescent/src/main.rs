//! Demo & micro-benchmark: build a k-NN graph three ways on the same data
//! and print the recall, latency, and distance-call count for each.
//!
//!     cargo run -p ruvector-nndescent --release
//!
//! Numbers from this binary are what the research README and ADR reference.

use ruvector_nndescent::{
    brute::BruteForce, nndescent::{NnDescent, NnDescentConfig},
    recall_at_k, KnnGraphBuilder, L2,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
fn gaussian_dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn run_nnd(label: &str, data: &[Vec<f32>], k: usize, cfg: NnDescentConfig, truth: &ruvector_nndescent::KnnGraph) {
    let r = NnDescent::new(L2, cfg).build(data, k);
    let recall = recall_at_k(truth, &r.graph);
    println!(
        "{:<16} | elapsed={:>8.3?} | dist_calls={:>10} | iters={} | recall={:.4}",
        label, r.elapsed, r.distance_calls, r.iterations, recall
    );
}

fn bench_one(label: &str, n: usize, d: usize, k: usize) {
    println!("\n=== dataset: {label}  N={n}  D={d}  K={k} ===");
    let data = gaussian_dataset(n, d, 0xABCD);
    let truth = BruteForce::new(L2).build(&data, k);
    println!(
        "brute-force      | elapsed={:>8.3?} | dist_calls={:>10} | iters={} | recall=1.0000",
        truth.elapsed, truth.distance_calls, truth.iterations
    );
    run_nnd("nnd-vanilla", &data, k,
        NnDescentConfig { rho: 1.0, reverse: false, ..Default::default() }, &truth.graph);
    run_nnd("nnd-reverse", &data, k,
        NnDescentConfig { rho: 1.0, reverse: true, ..Default::default() }, &truth.graph);
    run_nnd("nnd-reverse-r05", &data, k,
        NnDescentConfig { rho: 0.5, reverse: true, ..Default::default() }, &truth.graph);
}

fn main() {
    println!("ruvector-nndescent demo — building k-NN graphs three ways");
    // Three scales so trends are visible without exhausting the laptop.
    bench_one("small",  500,  32, 20);
    bench_one("medium", 2000, 64, 20);
    bench_one("large",  5000, 64, 20);
}
