//! Real benchmark for range-filtered ANN. No mocks, no aspirational numbers.
//!
//! Synthesises a deterministic SIFT-like workload (LCG-driven, 128-d) and
//! measures Recall@10 + per-query latency for three backends across three
//! selectivities (100%, 10%, 1%).
//!
//! Run:
//!     cargo run --release -p ruvector-serf --example range_bench

use ruvector_serf::{
    flat::Flat,
    nsw::NswParams,
    nsw_post::NswPost,
    recall,
    segment::SegmentGraph,
    Range, RangeAnn,
};
use std::sync::Arc;
use std::time::Instant;

const D: usize = 128;
const N: usize = 20_000;
const NQ: usize = 200;
const K: usize = 10;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}
fn lcg_f(seed: &mut u64) -> f32 {
    (lcg(seed) as f32) / (u32::MAX as f32)
}

fn make_vec(seed: &mut u64) -> Vec<f32> {
    (0..D).map(|_| lcg_f(seed) * 2.0 - 1.0).collect()
}

fn main() {
    println!("== ruvector-serf range-filtered ANN benchmark ==");
    println!("D={D}  N={N}  NQ={NQ}  K={K}\n");

    let mut s: u64 = 0xC0FFEE;
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| make_vec(&mut s)).collect();
    let keys: Vec<f32> = (0..N).map(|i| i as f32 / N as f32).collect(); // uniform [0,1)
    let queries: Vec<Vec<f32>> = (0..NQ).map(|_| make_vec(&mut s)).collect();

    let vectors_arc = Arc::new(vectors.clone());

    // ---------- Build phase ----------
    let t = Instant::now();
    let flat = Flat::new(vectors.clone(), keys.clone());
    let flat_build = t.elapsed();

    let params = NswParams {
        m: 16,
        ef_construction: 64,
        ef_search: 64,
    };

    let t = Instant::now();
    let nsw_post = NswPost::build(vectors_arc.clone(), keys.clone(), params, 8);
    let nsw_build = t.elapsed();
    let nsw_bytes = nsw_post.graph_bytes();

    let t = Instant::now();
    let segg = SegmentGraph::build(vectors_arc.clone(), keys.clone(), params, 256);
    let seg_build = t.elapsed();
    let seg_bytes = segg.graph_bytes();
    let seg_count = segg.num_graphs();

    println!("Build times:");
    println!("  flat            : {:>8.2?}", flat_build);
    println!("  nsw-postfilter  : {:>8.2?}   adj-bytes={}", nsw_build, nsw_bytes);
    println!(
        "  serf-segment   : {:>8.2?}   adj-bytes={}  graphs={}",
        seg_build, seg_bytes, seg_count
    );
    println!();

    // ---------- Query phase ----------
    let selectivities = [(1.00f32, "100%"), (0.10, "10%"), (0.01, "1%")];

    println!("{:<18} {:<10} {:>10} {:>14} {:>10}", "method", "sel", "recall@10", "us/query", "qps");
    println!("{}", "-".repeat(68));

    for (sel, sel_name) in selectivities {
        // Build per-query ranges that cover the target selectivity, centred at varying spots.
        let ranges: Vec<Range> = (0..NQ)
            .map(|i| {
                let c = (i as f32 + 0.5) / NQ as f32;
                let half = sel / 2.0;
                let lo = (c - half).max(0.0);
                let hi = (c + half).min(1.0);
                Range { lo, hi }
            })
            .collect();

        // Ground truth via Flat.
        let truths: Vec<Vec<(usize, f32)>> = queries
            .iter()
            .zip(ranges.iter())
            .map(|(q, r)| flat.search(q, *r, K))
            .collect();

        for backend in [&flat as &dyn RangeAnn, &nsw_post, &segg] {
            let t = Instant::now();
            let mut total_recall = 0.0f32;
            for (i, q) in queries.iter().enumerate() {
                let res = backend.search(q, ranges[i], K);
                total_recall += recall(&res, &truths[i]);
            }
            let elapsed = t.elapsed();
            let us = (elapsed.as_secs_f64() * 1e6) / NQ as f64;
            let qps = NQ as f64 / elapsed.as_secs_f64();
            println!(
                "{:<18} {:<10} {:>10.3} {:>14.1} {:>10.0}",
                backend.name(),
                sel_name,
                total_recall / NQ as f32,
                us,
                qps
            );
        }
        println!();
    }
}
