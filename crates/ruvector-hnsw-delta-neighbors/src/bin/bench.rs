//! Real cargo-run benchmark for delta-neighbor storage.
//!
//! Reports: bytes/node for each store, decode ns/list (median), and compression
//! ratio vs the flat baseline. Runs before and after locality remapping so the
//! effect of remap is visible in isolation.

use ruvector_hnsw_delta_neighbors::{
    apply_permutation, brute_knn_graph, locality_remap_bfs, measure_all,
    BitpackedDeltaStore, FlatU32Store, NeighborStore, VarintDeltaStore,
};
use ruvector_hnsw_delta_neighbors::graph::random_vectors;
use std::time::Instant;

fn median(vs: &mut Vec<u128>) -> u128 {
    vs.sort_unstable();
    vs[vs.len() / 2]
}

fn bench_decode<S: NeighborStore>(store: &S, n: usize) -> u128 {
    // Warm up.
    let mut buf = Vec::with_capacity(64);
    for i in 0..n.min(1000) {
        store.decode(i as u32, &mut buf);
    }
    // Time per-node decode; repeat multiple full passes.
    let passes = 8;
    let mut sample_ns: Vec<u128> = Vec::with_capacity(passes);
    for _ in 0..passes {
        let t0 = Instant::now();
        let mut acc: u32 = 0;
        for i in 0..n {
            store.decode(i as u32, &mut buf);
            // Prevent LLVM from eliminating the decode.
            if !buf.is_empty() {
                acc = acc.wrapping_add(buf[0]);
            }
        }
        let elapsed = t0.elapsed().as_nanos();
        std::hint::black_box(acc);
        sample_ns.push(elapsed / n as u128);
    }
    median(&mut sample_ns)
}

fn run(n: usize, d: usize, k: usize) {
    println!("\n=== n={} d={} k={} ===", n, d, k);
    let v = random_vectors(n, d, 42);
    let g_raw = brute_knn_graph(&v, n, d, k);

    let sz_raw = measure_all(&g_raw);
    println!("[before remap] flat={}B varint={}B ({:.2}x) bitpacked={}B ({:.2}x)",
        sz_raw.flat, sz_raw.varint, sz_raw.varint_ratio(),
        sz_raw.bitpacked, sz_raw.bitpacked_ratio());

    let perm = locality_remap_bfs(&g_raw, 0);
    let g = apply_permutation(&g_raw, &perm);
    let sz = measure_all(&g);
    println!("[after remap]  flat={}B varint={}B ({:.2}x) bitpacked={}B ({:.2}x)",
        sz.flat, sz.varint, sz.varint_ratio(),
        sz.bitpacked, sz.bitpacked_ratio());
    println!("bytes/node: flat={:.2} varint={:.2} bitpacked={:.2}",
        sz.flat as f64 / n as f64,
        sz.varint as f64 / n as f64,
        sz.bitpacked as f64 / n as f64);

    let s_flat = FlatU32Store::from_graph(&g);
    let s_var = VarintDeltaStore::from_graph(&g);
    let s_bp = BitpackedDeltaStore::from_graph(&g);

    let t_flat = bench_decode(&s_flat, n);
    let t_var = bench_decode(&s_var, n);
    let t_bp = bench_decode(&s_bp, n);
    println!("decode ns/list: flat={} varint={} bitpacked={}", t_flat, t_var, t_bp);
}

fn main() {
    println!("ruvector-hnsw-delta-neighbors benchmark");
    println!("(brute k-NN graph on random unit vectors; sorted neighbor lists)");
    run(2_000, 16, 16);
    run(5_000, 32, 32);
    run(10_000, 64, 32);
}
