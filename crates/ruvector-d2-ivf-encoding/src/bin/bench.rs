//! End-to-end benchmark for D2-IVF posting encoding.
//!
//! Generates a synthetic IVF layout (1M IDs across configurable clusters),
//! then measures encoded size and decode throughput for raw-u32,
//! varint-delta, and d2-varint. Numbers are collected into a table
//! printed to stdout and pasted into the research doc.

use std::time::Instant;

use ruvector_d2_ivf_encoding::{
    synth_postings, synth_postings_ap_biased, synth_postings_sparse_ap, D2Varint, PostingCodec,
    RawU32, VarintDelta,
};

fn bench_variant<C: PostingCodec>(codec: &C, postings: &[Vec<u32>]) -> (usize, f64, f64) {
    // 1. Encode once, record size.
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(postings.len());
    let t_enc = Instant::now();
    for p in postings {
        encoded.push(codec.encode(p));
    }
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    let total_bytes: usize = encoded.iter().map(|e| e.len()).sum();

    // 2. Decode a few passes for throughput.
    let passes = 5;
    let mut total_ids: usize = 0;
    let t_dec = Instant::now();
    for _ in 0..passes {
        for e in &encoded {
            let ids = codec.decode(e);
            total_ids += ids.len();
        }
    }
    let dec_s = t_dec.elapsed().as_secs_f64();
    let mids_per_s = (total_ids as f64 / dec_s) / 1e6;

    eprintln!(
        "[{:>14}] encode={:>7.1} ms  bytes={:>10}  decode={:>6.1} M-IDs/s",
        codec.name(),
        enc_ms,
        total_bytes,
        mids_per_s
    );

    (total_bytes, enc_ms, mids_per_s)
}

fn main() {
    let n_total: usize = 1_000_000;
    let configs = [
        ("1K clusters (large postings)", 1_024_usize),
        ("4K clusters (medium postings)", 4_096),
        ("16K clusters (small postings)", 16_384),
    ];

    println!("# D2-IVF posting encoding — real cargo bench numbers");
    println!("# n_total={} IDs per config, u32 range [0, n_total)", n_total);
    println!("# Measured on cargo run --release -p ruvector-d2-ivf-encoding --bin bench");
    println!();

    println!("====================================================");
    println!("PART A: UNIFORM-RANDOM assignment (worst-case for D2)");
    println!("====================================================");
    println!();
    for (label, nc) in configs {
        let postings = synth_postings(n_total, nc, 42);
        let mean_len = n_total as f64 / nc as f64;
        println!("## {} (mean posting len = {:.1})", label, mean_len);
        println!(
            "{:<14}  {:>12}  {:>8}  {:>12}  {:>10}",
            "codec", "bytes", "b/ID", "encode ms", "dec M-ids/s"
        );
        println!("{}", "-".repeat(64));

        let raw = bench_variant(&RawU32, &postings);
        let vd = bench_variant(&VarintDelta, &postings);
        let d2 = bench_variant(&D2Varint, &postings);

        for (name, (bytes, enc_ms, dec)) in [
            ("raw-u32", raw),
            ("varint-delta", vd),
            ("d2-varint", d2),
        ] {
            let b_per_id = bytes as f64 / n_total as f64;
            println!(
                "{:<14}  {:>12}  {:>8.2}  {:>12.1}  {:>10.1}",
                name, bytes, b_per_id, enc_ms, dec
            );
        }
        println!(
            "savings: d2 vs raw = {:.2}x   d2 vs varint-delta = {:.2}x",
            raw.0 as f64 / d2.0 as f64,
            vd.0 as f64 / d2.0 as f64
        );
        println!();
    }

    println!("====================================================");
    println!("PART B: AP-biased, dense slab (stride~1, locality reorder)");
    println!("====================================================");
    println!();
    // Jitter sweep: 0 = perfect AP, higher = more disorder.
    let jitter_configs = [
        ("jitter=0 (perfect AP)", 0i64),
        ("jitter=2", 2),
        ("jitter=8", 8),
        ("jitter=32", 32),
    ];
    for (label, jitter) in jitter_configs {
        let postings = synth_postings_ap_biased(n_total, 1024, jitter, 42);
        let actual_ids: usize = postings.iter().map(|p| p.len()).sum();
        println!("## {} — 1024 clusters, {} IDs (slab stride ~1)", label, actual_ids);
        println!(
            "{:<14}  {:>12}  {:>8}  {:>12}  {:>10}",
            "codec", "bytes", "b/ID", "encode ms", "dec M-ids/s"
        );
        println!("{}", "-".repeat(64));

        let raw = bench_variant(&RawU32, &postings);
        let vd = bench_variant(&VarintDelta, &postings);
        let d2 = bench_variant(&D2Varint, &postings);

        for (name, (bytes, enc_ms, dec)) in [
            ("raw-u32", raw),
            ("varint-delta", vd),
            ("d2-varint", d2),
        ] {
            let b_per_id = bytes as f64 / actual_ids as f64;
            println!(
                "{:<14}  {:>12}  {:>8.2}  {:>12.1}  {:>10.1}",
                name, bytes, b_per_id, enc_ms, dec
            );
        }
        println!(
            "savings: d2 vs raw = {:.2}x   d2 vs varint-delta = {:.2}x",
            raw.0 as f64 / d2.0 as f64,
            vd.0 as f64 / d2.0 as f64
        );
        println!();
    }

    println!("====================================================");
    println!("PART C: Sparse-ID AP regime (stride>128) — D2's win zone");
    println!("====================================================");
    println!();
    let stride_configs: &[(&str, i64, i64)] = &[
        ("stride=1000 jitter=5", 1000, 5),
        ("stride=1000 jitter=50", 1000, 50),
        ("stride=10000 jitter=5", 10_000, 5),
        ("stride=10000 jitter=500", 10_000, 500),
        ("stride=100000 jitter=50", 100_000, 50),
    ];
    for (label, stride, jitter) in stride_configs {
        let postings = synth_postings_sparse_ap(1000, 1024, *stride, *jitter, 42);
        let actual_ids: usize = postings.iter().map(|p| p.len()).sum();
        println!("## {} — 1024 clusters x 1000 IDs = {} IDs", label, actual_ids);
        println!(
            "{:<14}  {:>12}  {:>8}  {:>12}  {:>10}",
            "codec", "bytes", "b/ID", "encode ms", "dec M-ids/s"
        );
        println!("{}", "-".repeat(64));

        let raw = bench_variant(&RawU32, &postings);
        let vd = bench_variant(&VarintDelta, &postings);
        let d2 = bench_variant(&D2Varint, &postings);

        for (name, (bytes, enc_ms, dec)) in [
            ("raw-u32", raw),
            ("varint-delta", vd),
            ("d2-varint", d2),
        ] {
            let b_per_id = bytes as f64 / actual_ids as f64;
            println!(
                "{:<14}  {:>12}  {:>8.2}  {:>12.1}  {:>10.1}",
                name, bytes, b_per_id, enc_ms, dec
            );
        }
        println!(
            "savings: d2 vs raw = {:.2}x   d2 vs varint-delta = {:.2}x",
            raw.0 as f64 / d2.0 as f64,
            vd.0 as f64 / d2.0 as f64
        );
        println!();
    }
}
