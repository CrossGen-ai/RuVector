//! End-to-end benchmark driver.
//!
//! `cargo run --release -p ruvector-succinct-hnsw --bin benchmark`

use ruvector_succinct_hnsw::bench::{format_report, run, BenchConfig};

fn main() {
    println!("# ruvector-succinct-hnsw benchmark\n");
    let scenarios = [
        (
            "d32_clustered_5k",
            BenchConfig { dim: 32, n: 5_000, clustered: true, ..BenchConfig::default() },
        ),
        (
            "d64_clustered_5k",
            BenchConfig { dim: 64, n: 5_000, clustered: true, ..BenchConfig::default() },
        ),
        (
            "d32_isotropic_5k",
            BenchConfig { dim: 32, n: 5_000, clustered: false, ..BenchConfig::default() },
        ),
        (
            "d32_clustered_20k",
            BenchConfig { dim: 32, n: 20_000, clustered: true, ..BenchConfig::default() },
        ),
        (
            "d32_clustered_5k_m32",
            BenchConfig { dim: 32, n: 5_000, m: 32, ef_construction: 128, clustered: true, ..BenchConfig::default() },
        ),
    ];
    for (label, cfg) in scenarios {
        println!("## scenario: {label}\n");
        let r = run(cfg);
        println!("{}", format_report(&r));
    }
}
