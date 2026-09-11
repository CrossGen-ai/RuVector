//! End-to-end recall + throughput harness for the three encoders.
//!
//! Usage:
//!   cargo run --release -p ruvector-ternary --bin benchmark
//!
//! Prints a Markdown-friendly table with real numbers on the current machine.
//! Deterministic (seed = 42) so the research document reproduces exactly.

use std::time::Instant;

use rand_distr::{Distribution, StandardNormal};

use ruvector_ternary::binary::{BinaryDistance, BinaryEncoder};
use ruvector_ternary::int8::{Int8Distance, Int8Encoder};
use ruvector_ternary::ternary::{TernaryDistance, TernaryEncoder};
use ruvector_ternary::{l2, seeded_rng, Distance, Encoder};

#[derive(Clone, Copy)]
struct Cfg {
    n: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
    seed: u64,
    sparsity: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            n: 20_000,
            dim: 128,
            n_queries: 200,
            k: 10,
            seed: 42,
            sparsity: 0.5,
        }
    }
}

fn make_corpus(cfg: Cfg) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = seeded_rng(cfg.seed);
    let corpus: Vec<Vec<f32>> = (0..cfg.n)
        .map(|_| (0..cfg.dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..cfg.n_queries)
        .map(|_| (0..cfg.dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect();
    (corpus, queries)
}

fn oracle_topk(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2(v, q)))
        .collect();
    d.sort_by(|a, b| a.1.total_cmp(&b.1));
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn topk_by_code<E: Encoder, D: Distance<Code = E::Code>>(
    enc: &E,
    dist: &D,
    corpus_codes: &[E::Code],
    q_code: &E::Code,
    k: usize,
) -> Vec<usize> {
    let mut d: Vec<(usize, u32)> =
        corpus_codes.iter().enumerate().map(|(i, v)| (i, dist.dist(v, q_code))).collect();
    d.sort_by_key(|x| x.1);
    let _ = enc; // silence unused for E generic
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(pred: &[usize], truth: &[usize]) -> f32 {
    let mut hits = 0;
    for p in pred {
        if truth.contains(p) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

struct Report {
    name: &'static str,
    bytes_per_code: usize,
    encode_ns_per_vec: f64,
    scan_ns_per_pair: f64,
    scan_ns_per_query: f64,
    recall10: f32,
}

fn run<E, D>(name: &'static str, enc: E, dist: D, corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize, truth: &[Vec<usize>]) -> Report
where
    E: Encoder,
    D: Distance<Code = E::Code>,
{
    // Encode
    let t = Instant::now();
    let corpus_codes: Vec<E::Code> = corpus.iter().map(|v| enc.encode(v)).collect();
    let encode_ns_per_vec = t.elapsed().as_nanos() as f64 / corpus.len() as f64;

    let query_codes: Vec<E::Code> = queries.iter().map(|v| enc.encode(v)).collect();

    // Scan + recall
    let t = Instant::now();
    let mut r = 0f32;
    for (qi, q_code) in query_codes.iter().enumerate() {
        let pred = topk_by_code(&enc, &dist, &corpus_codes, q_code, k);
        r += recall_at_k(&pred, &truth[qi]);
    }
    let scan_elapsed = t.elapsed().as_nanos() as f64;
    let n_pairs = (queries.len() * corpus.len()) as f64;
    let scan_ns_per_pair = scan_elapsed / n_pairs;
    let scan_ns_per_query = scan_elapsed / queries.len() as f64;

    Report {
        name,
        bytes_per_code: enc.bytes_per_code(),
        encode_ns_per_vec,
        scan_ns_per_pair,
        scan_ns_per_query,
        recall10: r / queries.len() as f32,
    }
}

fn print_report(cfg: Cfg, reports: &[Report], truth_ns_per_query: f64, fp32_bytes: usize) {
    println!();
    println!("## Config");
    println!("- N = {}", cfg.n);
    println!("- dim = {}", cfg.dim);
    println!("- queries = {}", cfg.n_queries);
    println!("- k = {}", cfg.k);
    println!("- seed = {}", cfg.seed);
    println!("- ternary sparsity = {:.2}", cfg.sparsity);
    println!();
    println!("| Encoder | Bytes/vec | Compression | Encode ns/vec | Scan ns/pair | Scan ns/query | Recall@{} |", cfg.k);
    println!("|---------|-----------|-------------|---------------|--------------|---------------|-----------|");
    for r in reports {
        let compression = fp32_bytes as f64 / r.bytes_per_code as f64;
        println!(
            "| {} | {} | {:.1}x | {:.1} | {:.2} | {:.0} | {:.4} |",
            r.name, r.bytes_per_code, compression, r.encode_ns_per_vec, r.scan_ns_per_pair, r.scan_ns_per_query, r.recall10
        );
    }
    println!(
        "| fp32-oracle | {} | 1.0x | - | - | {:.0} | 1.0000 |",
        fp32_bytes, truth_ns_per_query
    );
    println!();
}

fn main() {
    let cfg = Cfg::default();
    println!("# ruvector-ternary benchmark");
    println!("Building corpus (N={}, dim={})...", cfg.n, cfg.dim);
    let (corpus, queries) = make_corpus(cfg);

    // Ground truth
    let t = Instant::now();
    let truth: Vec<Vec<usize>> = queries.iter().map(|q| oracle_topk(&corpus, q, cfg.k)).collect();
    let truth_ns = t.elapsed().as_nanos() as f64 / queries.len() as f64;

    let fp32_bytes = cfg.dim * 4;

    let mut reports = vec![];
    reports.push(run(
        "binary",
        BinaryEncoder::new(cfg.dim),
        BinaryDistance,
        &corpus,
        &queries,
        cfg.k,
        &truth,
    ));
    reports.push(run(
        "ternary",
        TernaryEncoder::new(cfg.dim, cfg.sparsity),
        TernaryDistance,
        &corpus,
        &queries,
        cfg.k,
        &truth,
    ));
    reports.push(run(
        "int8",
        Int8Encoder::new(cfg.dim),
        Int8Distance,
        &corpus,
        &queries,
        cfg.k,
        &truth,
    ));

    print_report(cfg, &reports, truth_ns, fp32_bytes);

    // Extra sweep on ternary sparsity, since the paper's headline claim is
    // "sparsity is the recall knob".
    println!("## Ternary sparsity sweep");
    println!("| Sparsity | Bytes/vec | Recall@{} | Scan ns/pair |", cfg.k);
    println!("|----------|-----------|-----------|--------------|");
    for s in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] {
        let cfg2 = Cfg { sparsity: s, ..cfg };
        let r = run(
            "ternary",
            TernaryEncoder::new(cfg2.dim, s),
            TernaryDistance,
            &corpus,
            &queries,
            cfg2.k,
            &truth,
        );
        println!(
            "| {:.2} | {} | {:.4} | {:.2} |",
            s, r.bytes_per_code, r.recall10, r.scan_ns_per_pair
        );
    }

    println!();
    println!("Done. All numbers on the current host, deterministic seed = {}.", cfg.seed);
}
