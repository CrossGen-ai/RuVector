//! End-to-end benchmark: clustered data, PQ-trained codebooks, baseline PQ ADC
//! scan vs dominant-skip PQ ADC scan. All numbers are real: timed with
//! `std::time::Instant` on actual cargo-run release builds.

use ruvector_pq_dominant_code_skip::*;
use std::time::Instant;

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let m: usize = std::env::var("M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let k_cent: usize = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let n_queries: usize = 200;
    let kk: usize = 10;
    let iters: usize = 10;
    let n_clusters: usize = 24;

    println!(
        "config: N={} dim={} M={} K={} n_queries={} k={} iters={} n_clusters={}",
        n, dim, m, k_cent, n_queries, kk, iters, n_clusters
    );

    let t0 = Instant::now();
    let data = clustered_dataset(n, dim, n_clusters, 0xC0DEBEEF);
    let queries = clustered_dataset(n_queries, dim, n_clusters, 0xABCD1234);
    println!("gen data: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let codebook = Codebook::train(&data, m, k_cent, iters, 0x5EED_1234);
    println!("train PQ: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let baseline = BaselinePq::build(&data, codebook.clone());
    println!("build baseline: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let skip = DominantSkipPq::build(&data, codebook.clone());
    println!("build dominant-skip: {:?}", t0.elapsed());

    let s = skip.stats();
    println!(
        "dominant-skip stats: skip_rate={:.3} bytes_baseline={} bytes_skip={}  savings={:.1}%",
        s.skip_rate,
        s.bytes_baseline,
        s.bytes_skip,
        100.0 * (1.0 - s.bytes_skip as f64 / s.bytes_baseline as f64),
    );

    // Ground truth via brute force.
    let t0 = Instant::now();
    let mut truth: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        truth.push(brute_force(&data, queries.row(qi), kk));
    }
    println!("brute force truth: {:?}", t0.elapsed());

    // Warmup (ensure both have cache-warm LUTs the same way).
    for qi in 0..5 {
        let _ = baseline.search(queries.row(qi), kk);
        let _ = skip.search(queries.row(qi), kk);
    }

    let mut rows = Vec::new();

    // Baseline brute force reference row.
    let t0 = Instant::now();
    let mut b_recall_sum = 0.0;
    for qi in 0..n_queries {
        let r = brute_force(&data, queries.row(qi), kk);
        b_recall_sum += recall_at_k(&truth[qi], &r);
    }
    let bf_us = t0.elapsed().as_secs_f64() * 1e6 / n_queries as f64;
    rows.push(BenchRow {
        name: "brute_force",
        per_query_us: bf_us,
        recall: b_recall_sum / n_queries as f64,
        bytes: (n * dim * 4) as u64,
    });

    let mut r_base = run_search(&baseline, &queries, &truth, kk);
    r_base.bytes = (baseline.codes.len()) as u64;
    rows.push(r_base);

    let mut r_skip = run_search(&skip, &queries, &truth, kk);
    r_skip.bytes = (skip.bitmap.len() + skip.exc_codes.len() + skip.offsets.len() * 4) as u64;
    rows.push(r_skip);

    println!();
    println!(
        "{:<22} {:>14} {:>10} {:>14} {:>14}",
        "variant", "us/query", "recall", "bytes", "speedup_vs_bf"
    );
    let bf = rows[0].per_query_us;
    for r in &rows {
        println!(
            "{:<22} {:>14.3} {:>10.3} {:>14} {:>13.2}x",
            r.name,
            r.per_query_us,
            r.recall,
            r.bytes,
            bf / r.per_query_us.max(1e-9),
        );
    }

    // Speedup PQ -> PQ-skip.
    let base_us = rows[1].per_query_us;
    let skip_us = rows[2].per_query_us;
    println!(
        "\npq_dominant_skip / pq_baseline speedup: {:.2}x",
        base_us / skip_us.max(1e-9)
    );
    // Equality check: both must give identical top-k (ADC is exact).
    let mut mismatches = 0usize;
    for qi in 0..n_queries {
        let a = baseline.search(queries.row(qi), kk);
        let b = skip.search(queries.row(qi), kk);
        let aset: std::collections::HashSet<u32> = a.iter().map(|x| x.0).collect();
        let bset: std::collections::HashSet<u32> = b.iter().map(|x| x.0).collect();
        if aset != bset {
            mismatches += 1;
        }
    }
    println!("baseline vs skip top-{} set mismatches: {} / {}", kk, mismatches, n_queries);
}
