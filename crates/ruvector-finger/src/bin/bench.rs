//! `cargo run --release -p ruvector-finger --bin finger-bench`
//!
//! Prints a table of variant × {time, recall, flops} for the
//! configured dataset. Used by the research doc to capture real
//! numbers — the binary is the source of truth, the markdown copies
//! whatever it printed.

use ruvector_finger::bench_harness::{run_full_bench, BenchConfig};

fn main() {
    let cfg = BenchConfig::default();
    eprintln!(
        "running FINGER bench: n={} d={} q={} k={}",
        cfg.n, cfg.d, cfg.q, cfg.k
    );
    let results = run_full_bench(&cfg);

    println!("variant                  recall@10   us/query   flops/est");
    println!("-----------------------  ---------  ---------  ----------");
    for r in &results {
        println!(
            "{:<23}  {:>9.4}  {:>9.1}  {:>10}",
            r.name, r.recall_at_k, r.mean_us_per_query, r.flops_per_estimate
        );
    }

    // Speedup vs exact baseline.
    let exact = results
        .iter()
        .find(|r| r.name == "exact-fp32")
        .map(|r| r.mean_us_per_query)
        .unwrap_or(1.0);
    println!();
    println!("variant                  speedup-vs-exact");
    for r in &results {
        let s = exact / r.mean_us_per_query;
        println!("{:<23}  {:>5.2}x", r.name, s);
    }
}
