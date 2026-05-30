//! mar-demo: real benchmark binary for Matryoshka Adaptive Retrieval.
//!
//! Generates a synthetic Matryoshka-shaped corpus, runs three retrievers,
//! reports recall@10 against the brute-force-full ground truth and per-
//! query latency. No mocks; all numbers come from this run.

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_matryoshka::{
    l2_normalize, recall_at_k, BruteForceFull, BruteForceLow, MatryoshkaAdaptive, Retriever,
};
use std::time::Instant;

const N: usize = 20_000;
const FULL_DIM: usize = 768;
const N_QUERIES: usize = 500;
const K: usize = 10;
const SEED: u64 = 0xC0DE_DEAD_BEEF_1234;

/// Build a corpus where information mass is concentrated in early dimensions
/// (Matryoshka-style decay): coordinate i drawn from N(0, sigma_i^2) where
/// sigma_i^2 = 1/(1 + alpha * i). This mimics the trained behavior of MRL
/// embeddings where prefixes remain meaningful.
fn synth_matryoshka_corpus(n: usize, dim: usize, alpha: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0f32; n * dim];
    let sigmas: Vec<f32> = (0..dim).map(|i| (1.0f32 / (1.0 + alpha * i as f32)).sqrt()).collect();
    let normals: Vec<Normal<f32>> = sigmas.iter().map(|&s| Normal::new(0.0, s).unwrap()).collect();
    for row in 0..n {
        let off = row * dim;
        for j in 0..dim {
            data[off + j] = normals[j].sample(&mut rng);
        }
        l2_normalize(&mut data[off..off + dim]);
    }
    data
}

fn gen_queries(n: usize, dim: usize, alpha: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigmas: Vec<f32> = (0..dim).map(|i| (1.0f32 / (1.0 + alpha * i as f32)).sqrt()).collect();
    let normals: Vec<Normal<f32>> = sigmas.iter().map(|&s| Normal::new(0.0, s).unwrap()).collect();
    (0..n)
        .map(|_| {
            let mut q: Vec<f32> = (0..dim).map(|j| normals[j].sample(&mut rng)).collect();
            l2_normalize(&mut q);
            q
        })
        .collect()
}

struct RunReport {
    name: String,
    bytes: usize,
    qps: f64,
    p50_us: f64,
    p99_us: f64,
    recall_at_10: f32,
}

fn time_search<R: Retriever>(
    r: &R,
    queries: &[Vec<f32>],
    k: usize,
    truth: &[Vec<(u32, f32)>],
    label: &str,
) -> RunReport {
    // Warm-up
    for q in queries.iter().take(10) {
        let _ = r.search(q, k).expect("warmup");
    }
    let mut lats_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f32> = Vec::with_capacity(queries.len());
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let res = r.search(q, k).expect("search");
        lats_us.push(t.elapsed().as_secs_f64() * 1e6);
        recalls.push(recall_at_k(&truth[qi], &res, k));
    }
    let total = t0.elapsed().as_secs_f64();
    lats_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lats_us[lats_us.len() / 2];
    let p99 = lats_us[(lats_us.len() as f64 * 0.99) as usize];
    let mean_recall: f32 = recalls.iter().sum::<f32>() / recalls.len() as f32;
    RunReport {
        name: label.to_string(),
        bytes: r.resident_bytes(),
        qps: queries.len() as f64 / total,
        p50_us: p50,
        p99_us: p99,
        recall_at_10: mean_recall,
    }
}

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 20 {
        format!("{:.2} MiB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.2} KiB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{} B", b)
    }
}

fn main() {
    let alpha: f32 = 0.02;
    println!("== ruvector-matryoshka MAR demo ==");
    println!(
        "corpus n={}, full_dim={}, queries={}, k={}, alpha={}",
        N, FULL_DIM, N_QUERIES, K, alpha
    );

    let t0 = Instant::now();
    let corpus = synth_matryoshka_corpus(N, FULL_DIM, alpha, SEED);
    let queries = gen_queries(N_QUERIES, FULL_DIM, alpha, SEED.wrapping_add(1));
    println!("synth corpus built in {:.2}s", t0.elapsed().as_secs_f64());

    let full = BruteForceFull::new(corpus.clone(), FULL_DIM).expect("full index");
    println!("full-index resident: {}", fmt_bytes(full.resident_bytes()));

    // Ground truth = brute force full.
    let t0 = Instant::now();
    let truth: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| full.search(q, K).expect("truth"))
        .collect();
    println!(
        "ground truth ({} queries) in {:.2}s",
        N_QUERIES,
        t0.elapsed().as_secs_f64()
    );

    let full_report = time_search(&full, &queries, K, &truth, "brute-full d=768");

    // Low-dim brute force at 64 — measure recall vs the *truncated* query.
    let low_dim = 64;
    let low = BruteForceLow::new(&corpus, FULL_DIM, low_dim).expect("low");
    // Need truncated+normalized queries.
    let q_low: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| {
            let mut p = q[..low_dim].to_vec();
            l2_normalize(&mut p);
            p
        })
        .collect();
    let low_report = time_search(&low, &q_low, K, &truth, "brute-low  d=64 ");

    // Matryoshka adaptive runs (full-dim queries).
    let configs = [(64usize, 4usize), (64, 8), (128, 4), (128, 16), (256, 2), (256, 8)];
    let mut mar_reports = Vec::new();
    for (ld, rf) in configs {
        let mar = MatryoshkaAdaptive::new(corpus.clone(), FULL_DIM, ld, rf).expect("mar");
        let lbl = format!("MAR        d={}/rerank x{}", ld, rf);
        mar_reports.push(time_search(&mar, &queries, K, &truth, &lbl));
    }

    println!("\n{:<28} | {:>12} | {:>10} | {:>10} | {:>10} | {:>10}",
        "method", "resident", "qps", "p50(us)", "p99(us)", "recall@10");
    println!("{}", "-".repeat(98));
    for r in std::iter::once(&full_report)
        .chain(std::iter::once(&low_report))
        .chain(mar_reports.iter())
    {
        println!(
            "{:<28} | {:>12} | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.4}",
            r.name,
            fmt_bytes(r.bytes),
            r.qps,
            r.p50_us,
            r.p99_us,
            r.recall_at_10
        );
    }

    // Acceptance numeric check: best MAR config should beat brute-low recall
    // and reach >= 0.95 recall@10 while using less wall time than brute-full.
    let best_mar = mar_reports
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
        .expect("at least one mar config");
    println!(
        "\nACCEPTANCE: best MAR = {} (recall@10={:.4}, qps={:.1}); brute-full qps={:.1}, brute-low recall@10={:.4}",
        best_mar.name, best_mar.recall_at_10, best_mar.qps, full_report.qps, low_report.recall_at_10
    );
    let mar_better = best_mar.recall_at_10 > low_report.recall_at_10;
    let mar_fast = best_mar.qps > full_report.qps;
    let mar_high = best_mar.recall_at_10 >= 0.95;
    println!(
        "  better-recall-than-low: {}, faster-than-full: {}, recall>=0.95: {}",
        mar_better, mar_fast, mar_high
    );
    if !(mar_better && mar_fast) {
        eprintln!("WARNING: MAR did not strictly dominate prefix-only on both axes for this corpus");
    }
}

