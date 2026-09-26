//! `cargo run --release --example run_bench` — quick sanity run against
//! a small dataset. Used for the research-doc "How it works" walkthrough.

use ruvector_hnsw_batch_shared_frontier::bench::run_bench;

fn main() {
    let r = run_bench(
        /* n */ 2000,
        /* dim */ 64,
        /* q */ 16,
        /* k */ 10,
        /* ef */ 48,
        /* correlation */ 0.9,
        /* seed */ 7,
    );
    r.print();
}
