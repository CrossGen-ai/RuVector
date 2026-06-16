//! Benchmark harness: compares four range-filtered ANN variants.
//!
//! - **brute**: linear scan over in-range points (ground truth, R=100%)
//! - **full-graph + post-filter**: full graph search with large beam, drop
//!   out-of-range results
//! - **full-graph + pre-filter**: predicate-agnostic expansion, only in-range
//!   nodes enter the result heap (ACORN-like)
//! - **segment-rangehnsw**: this crate — segment tree of mini-graphs
//!
//! Run with: `cargo run --release -p ruvector-segment-rangehnsw --example bench`.

use ruvector_segment_rangehnsw::{
    brute_force_range,
    dist::Lcg,
    graph::Graph,
    segment::SegmentRangeIndex,
};
use std::time::Instant;

fn gen_points(n: usize, d: usize, seed: u64) -> Vec<(u32, f32, Vec<f32>)> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..d).map(|_| rng.next_gauss()).collect();
            // key: uniform in [0, 1) — natural "selectivity = hi-lo"
            let key = rng.next_f32();
            (i as u32, key, v)
        })
        .collect()
}

fn gen_query_ranges(num: usize, sel: f32, seed: u64) -> Vec<(f32, f32)> {
    let mut rng = Lcg::new(seed);
    (0..num)
        .map(|_| {
            let center = rng.next_f32() * (1.0 - sel);
            (center, center + sel)
        })
        .collect()
}

fn recall(gt: &[u32], got: &[u32]) -> f32 {
    if gt.is_empty() { return 1.0; }
    let h = gt.iter().filter(|g| got.contains(g)).count() as f32;
    h / gt.len() as f32
}

fn bench_round(n: usize, d: usize, k: usize, sel: f32, queries: usize) {
    println!("\n=== n={} d={} k={} selectivity={:.2} queries={} ===", n, d, k, sel, queries);
    let pts = gen_points(n, d, 42);

    let t0 = Instant::now();
    let segment = SegmentRangeIndex::build(pts.clone(), 256, 16, 64);
    let seg_build = t0.elapsed().as_secs_f64() * 1000.0;

    // Build single flat graph over the whole dataset (sorted by key, but
    // searched as a regular ANN graph).
    let sorted = {
        let mut p = pts.clone();
        p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        p
    };
    let vectors: Vec<Vec<f32>> = sorted.iter().map(|(_, _, v)| v.clone()).collect();
    let gids: Vec<u32> = sorted.iter().map(|(g, _, _)| *g).collect();
    let keys: Vec<f32> = sorted.iter().map(|(_, k, _)| *k).collect();
    let t0 = Instant::now();
    let flat = Graph::build(vectors, gids, keys, 16, 64);
    let flat_build = t0.elapsed().as_secs_f64() * 1000.0;

    let mut rng = Lcg::new(7);
    let queries_q: Vec<Vec<f32>> = (0..queries).map(|_| (0..d).map(|_| rng.next_gauss()).collect()).collect();
    let ranges = gen_query_ranges(queries, sel, 13);

    // Ground truth.
    let mut gts: Vec<Vec<u32>> = Vec::with_capacity(queries);
    let t0 = Instant::now();
    for i in 0..queries {
        let (lo, hi) = ranges[i];
        let gt = brute_force_range(&pts, &queries_q[i], lo, hi, k);
        gts.push(gt.into_iter().map(|x| x.1).collect());
    }
    let brute_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Post-filter on full graph: search 8k, drop out-of-range.
    let ef = (8 * k).max(64);
    let mut post_recall = 0.0f32;
    let t0 = Instant::now();
    for i in 0..queries {
        let (lo, hi) = ranges[i];
        let mut hits = flat.search(&queries_q[i], ef);
        hits.retain(|(_, _, key)| (lo..=hi).contains(key));
        hits.truncate(k);
        let got: Vec<u32> = hits.into_iter().map(|x| x.1).collect();
        post_recall += recall(&gts[i], &got);
    }
    let post_ms = t0.elapsed().as_secs_f64() * 1000.0;
    post_recall /= queries as f32;

    // Pre-filter on full graph (ACORN-style).
    let mut pre_recall = 0.0f32;
    let t0 = Instant::now();
    for i in 0..queries {
        let (lo, hi) = ranges[i];
        let mut hits = flat.search_range_prefilter(&queries_q[i], ef, lo, hi);
        hits.truncate(k);
        let got: Vec<u32> = hits.into_iter().map(|x| x.1).collect();
        pre_recall += recall(&gts[i], &got);
    }
    let pre_ms = t0.elapsed().as_secs_f64() * 1000.0;
    pre_recall /= queries as f32;

    // Segment range index.
    let mut seg_recall = 0.0f32;
    let t0 = Instant::now();
    for i in 0..queries {
        let (lo, hi) = ranges[i];
        let hits = segment.search(&queries_q[i], k, lo, hi, 64);
        let got: Vec<u32> = hits.into_iter().map(|x| x.1).collect();
        seg_recall += recall(&gts[i], &got);
    }
    let seg_ms = t0.elapsed().as_secs_f64() * 1000.0;
    seg_recall /= queries as f32;

    let flat_mb = flat.memory_bytes() as f64 / (1024.0 * 1024.0);
    let seg_mb = segment.memory_bytes() as f64 / (1024.0 * 1024.0);

    println!("{:<28} {:>10}  {:>10}  {:>10}  {:>10}", "variant", "build_ms", "query_ms", "qps", "recall@k");
    let qps = |ms: f64| (queries as f64) * 1000.0 / ms;
    println!("{:<28} {:>10.1}  {:>10.2}  {:>10.0}  {:>10.3}", "brute_force_range", 0.0, brute_ms, qps(brute_ms), 1.000);
    println!("{:<28} {:>10.1}  {:>10.2}  {:>10.0}  {:>10.3}", "flat-graph + post-filter", flat_build, post_ms, qps(post_ms), post_recall);
    println!("{:<28} {:>10.1}  {:>10.2}  {:>10.0}  {:>10.3}", "flat-graph + pre-filter", flat_build, pre_ms, qps(pre_ms), pre_recall);
    println!("{:<28} {:>10.1}  {:>10.2}  {:>10.0}  {:>10.3}", "segment-rangehnsw", seg_build, seg_ms, qps(seg_ms), seg_recall);
    println!("memory: flat-graph={:.2} MiB, segment-rangehnsw={:.2} MiB", flat_mb, seg_mb);
}

fn main() {
    println!("ruvector-segment-rangehnsw bench — release build recommended");
    let n = 10_000;
    let d = 64;
    let k = 10;
    let queries = 100;
    for sel in &[0.01f32, 0.05, 0.20, 0.50] {
        bench_round(n, d, k, *sel, queries);
    }
}
