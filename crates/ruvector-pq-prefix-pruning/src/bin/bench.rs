//! Benchmark: Baseline vs PrefixPrune vs VarOrderedPrune.
//!
//! Reports mean per-query latency, total segment lookups, and group prune rate
//! across a 32-query workload on a synthetic clustered corpus.

use ruvector_pq_prefix_pruning::bench_util::make_corpus;
use ruvector_pq_prefix_pruning::{Baseline, PrefixPrune, QueryLut, Scanner, VarOrderedPrune};
use std::time::Instant;

fn run_scanner<S: Scanner>(
    mut s: S,
    codes: &ruvector_pq_prefix_pruning::PqCodes,
    queries: &[f32],
    book: &ruvector_pq_prefix_pruning::codec::Codebook,
    k: usize,
    repeats: usize,
) -> (f64, u64, u64, u64) {
    let d = book.params.d();
    let nq = queries.len() / d;
    let mut total_lookups = 0u64;
    let mut groups_pruned = 0u64;
    let mut groups_total = 0u64;
    let mut hits_sink = 0u64;
    // Warmup.
    for q in 0..nq {
        let qv = &queries[q * d..(q + 1) * d];
        let lut = QueryLut::build(book, qv);
        let (hits, _) = s.scan(codes, &lut, k);
        hits_sink = hits_sink.wrapping_add(hits[0].id as u64);
    }
    let t = Instant::now();
    for _ in 0..repeats {
        for q in 0..nq {
            let qv = &queries[q * d..(q + 1) * d];
            let lut = QueryLut::build(book, qv);
            let (hits, stats) = s.scan(codes, &lut, k);
            total_lookups += stats.seg_lookups;
            groups_pruned += stats.groups_pruned;
            groups_total += stats.groups_total;
            hits_sink = hits_sink.wrapping_add(hits[0].id as u64);
        }
    }
    let elapsed = t.elapsed().as_secs_f64();
    let per_query_us = elapsed * 1e6 / (repeats * nq) as f64;
    std::hint::black_box(hits_sink);
    (per_query_us, total_lookups, groups_pruned, groups_total)
}

fn main() {
    let cases = [
        // (n, m, d, nc, k, label)
        (100_000usize, 16usize, 128usize, 256usize, 10usize, "100k·m=16·d=128·k=10"),
        (100_000, 32, 128, 256, 10, "100k·m=32·d=128·k=10"),
        (100_000, 16, 128, 256, 100, "100k·m=16·d=128·k=100"),
        (500_000, 16, 128, 1024, 10, "500k·m=16·d=128·k=10"),
    ];
    let repeats = 3;
    println!(
        "{:<36} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "case", "scanner", "us/query", "lookups", "prune%", "speedup", "groups"
    );
    for (n, m, d, nc, k, label) in cases {
        let (_vecs, queries, book, codes) = make_corpus(n, m, d, nc, 0xC0FFEE);

        let (base_us, base_lookups, _, base_groups) =
            run_scanner(Baseline, &codes, &queries, &book, k, repeats);
        let base_lookups_per_q = base_lookups as f64 / (repeats * queries.len() / d) as f64;
        println!(
            "{:<36} {:>12} {:>12.2} {:>12.0} {:>10} {:>10} {:>10}",
            label, "baseline", base_us, base_lookups_per_q, "-", "1.00x", base_groups
        );

        for &(scanner_name, scanner_fn) in &[
            ("prefix_prune", 1usize),
            ("var_ordered", 2usize),
        ] {
            let (us, lookups, pruned, total) = if scanner_fn == 1 {
                run_scanner(PrefixPrune, &codes, &queries, &book, k, repeats)
            } else {
                run_scanner(VarOrderedPrune, &codes, &queries, &book, k, repeats)
            };
            let per_q_lookups = lookups as f64 / (repeats * queries.len() / d) as f64;
            let prune_pct = if total > 0 {
                100.0 * pruned as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "{:<36} {:>12} {:>12.2} {:>12.0} {:>9.1}% {:>9.2}x {:>10}",
                label,
                scanner_name,
                us,
                per_q_lookups,
                prune_pct,
                base_us / us,
                total
            );
        }
        println!();
    }
}
