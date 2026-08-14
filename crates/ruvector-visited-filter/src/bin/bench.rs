//! Real benchmark for the visited-filter backends.
//!
//! Generates a synthetic HNSW-like visit stream (uniform + skewed hub
//! traffic), runs each backend for a fixed workload, and reports total
//! elapsed nanoseconds, ns/insert, and searches/second.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_visited_filter::{
    BitmapVisited, GenerationVisited, HashSetVisited, SearchScratch, VisitedFilter,
};
use std::time::Instant;

/// One (num_nodes, visits_per_search) workload row.
struct Workload {
    label: &'static str,
    nodes: u32,
    per_search: usize,
    hub_fraction: f64,     // fraction of visits directed at a small hot set
    searches: usize,
}

fn build_streams(w: &Workload, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let hub_size = ((w.nodes as f64) * 0.005).max(8.0) as u32; // top 0.5% "hubs"
    (0..w.searches)
        .map(|_| {
            let mut s = Vec::with_capacity(w.per_search);
            for _ in 0..w.per_search {
                let u = if rng.gen_bool(w.hub_fraction) {
                    rng.gen_range(0..hub_size)
                } else {
                    rng.gen_range(0..w.nodes)
                };
                s.push(u);
            }
            s
        })
        .collect()
}

fn run<F: VisitedFilter>(
    label: &str, mut scratch: SearchScratch<F>, streams: &[Vec<u32>],
) -> (u128, u64, usize) {
    // Warm one search — matches production behavior where the scratch is
    // pooled across queries.
    scratch.simulate(streams[0].iter().copied());
    let bytes = scratch.filter.bytes();
    let total_ops: u64 = streams.iter().map(|s| s.len() as u64).sum();
    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for s in streams { sink ^= scratch.simulate(s.iter().copied()); }
    let elapsed = t0.elapsed().as_nanos();
    // Prevent DCE.
    std::hint::black_box(sink);
    println!(
        "  {:12} elapsed_ns={:>12}  ops={:>10}  ns/op={:>7.2}  \
         searches/s={:>10.0}  mem_KiB={:>7.1}",
        label,
        elapsed,
        total_ops,
        elapsed as f64 / total_ops as f64,
        streams.len() as f64 / (elapsed as f64 / 1.0e9),
        bytes as f64 / 1024.0,
    );
    (elapsed, total_ops, bytes)
}

fn main() {
    println!("ruvector-visited-filter — real benchmark");
    println!("build={} rustc={}", env!("CARGO_PKG_VERSION"), option_env!("RUSTC_VERSION").unwrap_or("stable"));

    let workloads = [
        Workload {
            label: "small-dense (100k nodes, ef=64, 20k queries, hub=0.30)",
            nodes: 100_000, per_search: 64, hub_fraction: 0.30, searches: 20_000,
        },
        Workload {
            label: "medium (1M nodes, ef=128, 5k queries, hub=0.20)",
            nodes: 1_000_000, per_search: 128, hub_fraction: 0.20, searches: 5_000,
        },
        Workload {
            label: "large-sparse (10M nodes, ef=256, 1k queries, hub=0.05)",
            nodes: 10_000_000, per_search: 256, hub_fraction: 0.05, searches: 1_000,
        },
    ];

    for w in &workloads {
        println!("\n== {} ==", w.label);
        let streams = build_streams(w, 0xC0FFEE);
        run("hashset",    SearchScratch::new(HashSetVisited::new()),           &streams);
        run("bitmap",     SearchScratch::new(BitmapVisited::new(w.nodes)),      &streams);
        run("generation", SearchScratch::new(GenerationVisited::new(w.nodes)),  &streams);
    }
}
