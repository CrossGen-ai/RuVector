//! Measured HNSW-adjacency codec comparison.
//!
//! Builds a synthetic graph (N nodes × degree D neighbors, ids in [0, N)),
//! compresses the adjacency with each codec, and reports:
//!   - total bytes
//!   - bytes/edge
//!   - compression ratio vs plain
//!   - decode throughput over a traversal workload

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_streamvbyte_hnsw::{
    AdjacencyCodec, CompressedAdjacency, DeltaVarintCodec, PlainCodec, StreamVByteCodec,
};
use std::time::Instant;

fn build_graph(n: usize, degree: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<u32> =
                (0..degree).map(|_| rng.gen_range(0..n as u32)).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect()
}

fn run_codec<C: AdjacencyCodec>(
    codec: C,
    lists: &[Vec<u32>],
    plain_bytes: usize,
    traversal: &[u32],
    total_edges_per_pass: usize,
) -> (String, usize, f64, f64) {
    let name = codec.name().to_string();
    let adj = CompressedAdjacency::build(codec, lists);
    let total = adj.total_bytes();
    let ratio = plain_bytes as f64 / total as f64;

    let mut scratch: Vec<u32> = Vec::with_capacity(64);
    let mut sink: u64 = 0;
    let iters = 3usize;
    let start = Instant::now();
    for _ in 0..iters {
        for &node in traversal {
            adj.neighbors(node as usize, &mut scratch);
            for &x in &scratch { sink = sink.wrapping_add(x as u64); }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    let edges_per_sec = (total_edges_per_pass * iters) as f64 / elapsed;
    (name, total, ratio, edges_per_sec)
}

fn main() {
    let configs = [
        ("small",  10_000usize,  16usize),
        ("medium", 100_000,       32),
        ("large",  500_000,       32),
    ];

    for &(label, n, degree) in &configs {
        println!("\n=== {} graph: N={}, degree={} ===", label, n, degree);
        let lists = build_graph(n, degree, 2026);
        let mean_deg: f64 =
            lists.iter().map(|l| l.len()).sum::<usize>() as f64 / lists.len() as f64;
        println!("  mean effective degree: {:.2}", mean_deg);

        let plain = CompressedAdjacency::build(PlainCodec, &lists);
        let plain_bytes = plain.total_bytes();

        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let traversal: Vec<u32> =
            (0..20_000).map(|_| rng.gen_range(0..n as u32)).collect();

        let total_edges_per_pass: usize = traversal.iter()
            .map(|&idx| lists[idx as usize].len())
            .sum();

        let results = vec![
            run_codec(PlainCodec,       &lists, plain_bytes, &traversal, total_edges_per_pass),
            run_codec(DeltaVarintCodec, &lists, plain_bytes, &traversal, total_edges_per_pass),
            run_codec(StreamVByteCodec, &lists, plain_bytes, &traversal, total_edges_per_pass),
        ];

        let edges_in_graph = lists.iter().map(|l| l.len()).sum::<usize>() as f64;

        println!(
            "  {:<22} {:>12} {:>10} {:>8} {:>18}",
            "codec", "bytes", "B/edge", "ratio", "edges/sec"
        );
        for (name, bytes, ratio, eps) in &results {
            let bpe = *bytes as f64 / edges_in_graph;
            println!(
                "  {:<22} {:>12} {:>10.3} {:>7.2}x {:>18.3e}",
                name, bytes, bpe, ratio, eps
            );
        }
    }
}
