//! Real-numbers report for the hamming-cascade PoC.
//!
//! Emits a markdown-friendly table of (recall@10, mean-latency, footprint)
//! for three cascade configurations against a synthetic Gaussian workload.
//! No mocks — every number is produced by running the actual code paths.

use std::collections::HashSet;
use std::time::Instant;

use ruvector_hamming_cascade::{
    Cascade, CascadeConfig, DistanceOracle, Fp32Oracle, HammingOracle, Int8Oracle,
};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // xorshift64 → uniform in [-1, 1]; not Gaussian but sufficient for a
    // reproducible relative comparison. All configs see the same data.
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn ground_truth(db: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    let mut o = Fp32Oracle::from_vectors(db);
    queries.iter().map(|q| {
        o.prime(q);
        let mut all: Vec<(f32, u32)> =
            (0..db.len()).map(|i| (o.score(i), i as u32)).collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.into_iter().take(k).map(|(_, i)| i).collect()
    }).collect()
}

fn recall_at_k(hits: &[u32], gt: &[u32]) -> f32 {
    let want: HashSet<u32> = gt.iter().copied().collect();
    hits.iter().filter(|h| want.contains(h)).count() as f32 / gt.len() as f32
}

fn run<C, F>(name: &str, mut casc: Cascade<C, F>, queries: &[Vec<f32>],
             gt: &[Vec<u32>]) -> (f32, f32, usize)
where C: DistanceOracle, F: DistanceOracle,
{
    // warm up
    for q in queries.iter().take(3) { let _ = casc.search(q); }
    let footprint = casc.footprint_bytes();
    let t0 = Instant::now();
    let mut recall_sum = 0.0f32;
    for (q, gt_q) in queries.iter().zip(gt.iter()) {
        let hits = casc.search(q);
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        recall_sum += recall_at_k(&ids, gt_q);
    }
    let elapsed = t0.elapsed();
    let mean_us = elapsed.as_secs_f64() * 1e6 / queries.len() as f64;
    let recall = recall_sum / queries.len() as f32;
    println!(
        "| {:20} | {:>7.3} | {:>10.2} | {:>14} |",
        name, recall, mean_us, format_bytes(footprint)
    );
    (recall, mean_us as f32, footprint)
}

fn format_bytes(b: usize) -> String {
    if b >= 1 << 20 { format!("{:.2} MiB", b as f64 / (1 << 20) as f64) }
    else if b >= 1 << 10 { format!("{:.2} KiB", b as f64 / (1 << 10) as f64) }
    else { format!("{b} B") }
}

fn main() {
    // Config: keep it small enough to run in CI, big enough to be non-trivial.
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let nq: usize = std::env::var("NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let k: usize = 10;
    let probe_k: usize = std::env::var("PROBE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(100);

    println!("# ruvector-hamming-cascade — real numbers");
    println!();
    println!("N={n} dim={dim} nq={nq} k={k} probe_k={probe_k}");
    println!();
    println!("| Config               | Recall  | µs/query   | Footprint      |");
    println!("|----------------------|---------|------------|----------------|");

    let db = synth(n, dim, 0xC0FFEE_1234_5678);
    let queries = synth(nq, dim, 0xBEEF_0000_1111_2222);
    let gt = ground_truth(&db, &queries, k);

    // Baseline: FP32 flat scan (probe_k = k, no rerank, both oracles fp32).
    let baseline = Cascade::new(
        Fp32Oracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k, probe_k: k },
    );
    run("fp32 flat (baseline)", baseline, &queries, &gt);

    // Alternative 1: INT8 coarse -> FP32 rerank.
    let int8 = Cascade::new(
        Int8Oracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k, probe_k },
    );
    run("int8 -> fp32 rerank", int8, &queries, &gt);

    // Alternative 2: Hamming coarse -> FP32 rerank.
    let hcasc = Cascade::new(
        HammingOracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k, probe_k },
    );
    run("hamming -> fp32 rerank", hcasc, &queries, &gt);

    // Alternative 3: Hamming-only (no rerank) — shows the recall cost.
    let honly = Cascade::new(
        HammingOracle::from_vectors(&db),
        HammingOracle::from_vectors(&db),
        CascadeConfig { k, probe_k: k },
    );
    run("hamming-only (no rerank)", honly, &queries, &gt);

    println!();
    println!("Acceptance target: hamming->fp32 recall >= 0.90 within 30% of FP32 latency.");
}
