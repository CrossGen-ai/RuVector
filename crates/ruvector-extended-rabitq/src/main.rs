//! `erabitq-demo` — end-to-end recall + throughput benchmark.
//!
//! Runs 1/2/4-bit Extended RaBitQ vs an f32 flat baseline on a synthetic
//! Gaussian corpus and prints a table. Emits JSON if `--json` is passed so
//! the research doc can ingest real numbers directly.

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use ruvector_extended_rabitq::{AnnIndex, ExtendedRabitqIndex, FlatF32Index};

fn gauss_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = vec![0f32; dim];
        for d in 0..dim {
            let x: f64 = rng.sample(StandardNormal);
            v[d] = x as f32;
        }
        out.push(v);
    }
    out
}

#[derive(Debug)]
struct Row {
    bits: u32,
    n: usize,
    dim: usize,
    code_bytes: usize,
    recall_at_10: f32,
    qps: f32,
    build_ms: f32,
}

fn run(n: usize, dim: usize, n_queries: usize, k: usize) -> Vec<Row> {
    let corpus = gauss_corpus(n, dim, 12345);
    let queries = gauss_corpus(n_queries, dim, 67890);

    let flat = FlatF32Index::from_vectors(dim, &corpus).unwrap();
    // Ground truth top-k.
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| flat.search(q, k).unwrap().into_iter().map(|r| r.id).collect())
        .collect();

    let mut rows = Vec::new();
    for &bits in &[1u32, 2, 4] {
        let t0 = Instant::now();
        let idx = ExtendedRabitqIndex::build(dim, bits, 42, &corpus).unwrap();
        let build_ms = t0.elapsed().as_secs_f32() * 1e3;

        let t0 = Instant::now();
        let mut hits = 0usize;
        let mut total = 0usize;
        for (q, ground) in queries.iter().zip(gt.iter()) {
            let got = idx.search(q, k).unwrap();
            let gt_set: std::collections::HashSet<u32> = ground.iter().copied().collect();
            for r in &got {
                if gt_set.contains(&r.id) {
                    hits += 1;
                }
            }
            total += k;
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let qps = n_queries as f32 / elapsed.max(1e-6);
        let recall = hits as f32 / total as f32;
        rows.push(Row {
            bits,
            n,
            dim,
            code_bytes: idx.code_bytes(),
            recall_at_10: recall,
            qps,
            build_ms,
        });
    }
    rows
}

fn main() {
    let json_out = env::args().any(|a| a == "--json");
    let scales = [(1_000usize, 128usize), (10_000, 128), (50_000, 128)];
    let mut all: Vec<Row> = Vec::new();
    for &(n, dim) in &scales {
        let rows = run(n, dim, 200, 10);
        all.extend(rows);
    }
    if json_out {
        // Minimal hand-rolled JSON so we don't need serde on Row.
        print!("[");
        for (i, r) in all.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"bits\":{},\"n\":{},\"dim\":{},\"code_bytes\":{},\"recall_at_10\":{:.4},\"qps\":{:.1},\"build_ms\":{:.2}}}",
                r.bits, r.n, r.dim, r.code_bytes, r.recall_at_10, r.qps, r.build_ms
            );
        }
        println!("]");
    } else {
        println!(
            "{:>4} {:>7} {:>4} {:>10} {:>10} {:>12} {:>10}",
            "bits", "n", "dim", "code_bytes", "recall@10", "qps", "build_ms"
        );
        for r in &all {
            println!(
                "{:>4} {:>7} {:>4} {:>10} {:>10.4} {:>12.1} {:>10.2}",
                r.bits, r.n, r.dim, r.code_bytes, r.recall_at_10, r.qps, r.build_ms
            );
        }
    }
}
