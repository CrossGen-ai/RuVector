//! Benchmark binary. Runs three modes and prints a compact table.

use ruvector_bidir_hnsw::bench::{default_config, run_bench};

fn main() {
    let base = default_config();
    println!(
        "bidir-hnsw bench  n={} dim={} k={} m_max={} ef_c={} r_max={} clusters={}",
        base.n, base.dim, base.k, base.m_max, base.ef_construction, base.r_max, base.n_clusters
    );
    println!(
        "{:<20}  {:>4}  {:>10}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}",
        "mode", "ef", "recall@10", "dist_evals", "nodes_visit", "qps", "build_s", "extra_B/n"
    );
    for &ef in &[16usize, 32, 64, 128, 256] {
        let mut cfg = default_config();
        cfg.ef = ef;
        let results = run_bench(&cfg);
        for r in &results {
            println!(
                "{:<20}  {:>4}  {:>10.4}  {:>12.1}  {:>12.1}  {:>10.1}  {:>10.3}  {:>10}",
                r.mode,
                ef,
                r.recall_at_k,
                r.avg_dist_evals,
                r.avg_nodes_visited,
                r.qps,
                r.build_seconds,
                r.extra_bytes_per_node
            );
        }
    }
}
