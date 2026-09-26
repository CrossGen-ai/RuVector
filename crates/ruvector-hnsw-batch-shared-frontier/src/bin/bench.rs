//! Bench binary. Runs a sweep of correlation strengths and batch sizes so
//! numbers reported in the ADR / research doc come from real cargo output.

use ruvector_hnsw_batch_shared_frontier::bench::run_bench;

fn main() {
    let dim = 128;
    let n = 5000;
    let k = 10;
    let ef = 64;

    println!("# ruvector-hnsw-batch-shared-frontier bench");
    println!("# dim={dim} n={n} k={k} ef={ef}");

    for &q in &[4usize, 8, 16, 32] {
        for &corr in &[0.6f32, 0.85, 0.98] {
            let r = run_bench(n, dim, q, k, ef, corr, 424242);
            r.print();
            println!();
        }
    }
}
