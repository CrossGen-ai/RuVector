//! Hand-rolled benchmark binary — no criterion dep, no mocks: times real
//! BFS traversals across the three backends and prints a table.

use std::time::Instant;

use ruvector_ef_hnsw_adjacency::{
    bfs_touch, build_random_graph, ef::EFAdjacency, fill_from, packed::PackedAdjacency,
    plain::PlainAdjacency, Adjacency,
};

fn bench_backend<A: Adjacency>(name: &str, adj: &A, iters: usize, budget: usize) -> f64 {
    // Warm.
    let mut total: u64 = 0;
    for i in 0..32 {
        total = total.wrapping_add(bfs_touch(adj, i as u32, 32));
    }
    std::hint::black_box(total);

    let start = Instant::now();
    let mut sink: u64 = 0;
    for i in 0..iters {
        let entry = (i as u32) % (adj.len() as u32);
        sink = sink.wrapping_add(bfs_touch(adj, entry, budget));
    }
    let elapsed = start.elapsed();
    std::hint::black_box(sink);
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    println!("  {:<8} {:>9.1} ns/traversal", name, ns);
    ns
}

fn main() {
    let configs: &[(usize, usize, usize, usize)] = &[
        // (n, m, iters, budget)
        (10_000, 16, 2_000, 64),
        (10_000, 32, 2_000, 64),
        (100_000, 16, 500, 64),
        (100_000, 32, 500, 64),
    ];

    println!("# ef-hnsw-adjacency benchmark");
    println!("# host: {}", std::env::consts::OS);
    for &(n, m, iters, budget) in configs {
        println!("\n== n = {}, m = {}, iters = {}, budget = {} ==", n, m, iters, budget);
        let g = build_random_graph(n, m, 42);
        let mut plain = PlainAdjacency::new();
        let mut packed = PackedAdjacency::new(n);
        let mut ef = EFAdjacency::new(n);
        fill_from(&mut plain, &g);
        fill_from(&mut packed, &g);
        fill_from(&mut ef, &g);

        let base = plain.bytes();
        println!(
            "  bytes: plain = {}, packed = {} ({:.2}x), ef = {} ({:.2}x)",
            base,
            packed.bytes(),
            base as f64 / packed.bytes() as f64,
            ef.bytes(),
            base as f64 / ef.bytes() as f64,
        );

        let _ = bench_backend("plain", &plain, iters, budget);
        let _ = bench_backend("packed", &packed, iters, budget);
        let _ = bench_backend("ef", &ef, iters, budget);
    }
}
