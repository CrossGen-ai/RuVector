//! Real benchmark for the three neighbor-store backends.
//!
//! Runs on synthetic HNSW-like graphs of configurable size/degree, and prints
//! bytes-per-edge, build time, and per-node decode latency.  No mocks — every
//! number in the research doc must be reproducible via `cargo run --release
//! -p ruvector-bitpacked-hnsw --bin bpk-bench`.

use ruvector_bitpacked_hnsw::{
    BitPackedStore, DeltaVarintStore, NeighborStore, RawU32Store,
};
use std::time::Instant;

fn synth_graph(n: usize, m: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut s = seed;
    let mut step = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    (0..n)
        .map(|_| (0..m).map(|_| (step() as u32) % (n as u32)).collect())
        .collect()
}

fn bench<S: NeighborStore>(name: &str, g: &[Vec<u32>], iters: usize) {
    let t0 = Instant::now();
    let store = S::build(g);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Full-graph scan × iters to smooth timer noise.
    let mut out = Vec::with_capacity(64);
    let mut sink: u64 = 0;
    let t1 = Instant::now();
    for _ in 0..iters {
        for i in 0..store.len() {
            store.decode(i as u32, &mut out);
            for &x in &out {
                sink = sink.wrapping_add(x as u64);
            }
        }
    }
    let elapsed = t1.elapsed().as_secs_f64();
    let total_decodes = (store.len() * iters) as f64;
    let ns_per_decode = elapsed * 1e9 / total_decodes;

    let edges = g.iter().map(|l| l.len()).sum::<usize>();
    let bpe = store.bytes() as f64 / edges as f64;

    println!(
        "{:<16}  bytes={:>10}  bpe={:>6.2}  build_ms={:>8.2}  decode_ns={:>7.1}  sink={}",
        name,
        store.bytes(),
        bpe,
        build_ms,
        ns_per_decode,
        sink & 0xFF, // suppress dead-code elim
    );
}

fn main() {
    let configs = [
        ("1k×16", 1_000, 16),
        ("10k×16", 10_000, 16),
        ("100k×16", 100_000, 16),
        ("10k×32", 10_000, 32),
        ("10k×64", 10_000, 64),
    ];

    for (label, n, m) in configs {
        println!("\n== graph {label} (N={n}, M={m}) ==");
        let g = synth_graph(n, m, 0xC0FFEE ^ (n as u64));
        // fewer scans for the big graph so the run stays under a minute
        let iters = if n >= 100_000 { 2 } else { 20 };
        bench::<RawU32Store>("raw-u32", &g, iters);
        bench::<DeltaVarintStore>("delta-varint", &g, iters);
        bench::<BitPackedStore>("bit-packed", &g, iters);
    }
}
