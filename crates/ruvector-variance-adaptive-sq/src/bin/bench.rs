//! VA-SQ benchmark binary. Produces recall@10 and per-query latency for
//! three variants under the SAME average bit budget per dim:
//!   1. SQ-uniform (baseline)
//!   2. VA-SQ linear-variance (ablation)
//!   3. VA-SQ log-variance (proposed)
//!
//! Also runs full-precision brute force to establish ground truth.
//!
//! Output is a text table printed to stdout; the research document copies
//! these numbers verbatim. No mocks: cargo run --release prints real wall
//! times from `std::time::Instant`.

use std::time::Instant;
use vasq::{
    brute_force_topk, gen_lognormal_embeddings, recall_at_k, AllocPolicy, VaSqCodec, VaSqConfig,
};

#[derive(Clone)]
struct Variant {
    name: &'static str,
    budget_bits: u32,
    b_min: u32,
    b_max: u32,
    policy: AllocPolicy,
}

fn run_variant(
    v: &Variant,
    dim: usize,
    train: &[f32],
    base: &[f32],
    queries: &[f32],
    gt: &[Vec<u32>],
    k: usize,
) -> (f32, f64, usize, f32) {
    let t_train = Instant::now();
    let codec = VaSqCodec::train(
        train,
        VaSqConfig {
            dim,
            budget_bits_per_dim: v.budget_bits,
            b_min: v.b_min,
            b_max: v.b_max,
            policy: v.policy,
        },
    );
    let _train_us = t_train.elapsed().as_micros();

    let codes = codec.encode(base);
    let lut = codec.build_lut();
    let n_base = base.len() / dim;
    let n_q = queries.len() / dim;
    let bpv = codec.bytes_per_vec();

    let t0 = Instant::now();
    let mut preds: Vec<Vec<u32>> = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let qv = &queries[q * dim..(q + 1) * dim];
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for i in 0..n_base {
            let code = &codes[i * bpv..(i + 1) * bpv];
            let d = codec.distance_lut(qv, code, &lut);
            if heap.len() < k {
                heap.push((d, i as u32));
                heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            } else if d < heap[k - 1].0 {
                heap[k - 1] = (d, i as u32);
                heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        preds.push(heap.into_iter().map(|(_, i)| i).collect());
    }
    let elapsed = t0.elapsed();
    let per_q_us = elapsed.as_micros() as f64 / n_q as f64;

    let recall = recall_at_k(gt, &preds, k);
    let bytes_total = n_base * bpv;
    let avg_bits = codec.average_bits_per_dim();
    (recall, per_q_us, bytes_total, avg_bits)
}

fn main() {
    let dim = 128;
    let n_train = 2_000;
    let n_base = 10_000;
    let n_query = 200;
    let k = 10;
    let seed = 20261028;

    println!("# VA-SQ Benchmark (dim={}, |base|={}, |queries|={}, k={})", dim, n_base, n_query, k);
    println!("# RNG seed: {}", seed);

    let ds = gen_lognormal_embeddings(dim, n_train, n_base, n_query, 0.9, seed);
    println!(
        "# Dataset: log-normal sigma profile, min_var={:.4}, max_var={:.4}, median_var={:.4}",
        min_f32(&ds.variances),
        max_f32(&ds.variances),
        median_f32(&ds.variances),
    );

    let t_gt = Instant::now();
    let gt = brute_force_topk(&ds.queries, &ds.base, dim, k);
    let gt_us = t_gt.elapsed().as_micros() as f64 / n_query as f64;
    println!("# Full-precision brute force: {:.1} us/query (baseline recall = 1.000)", gt_us);
    println!();

    let variants = [
        Variant {
            name: "SQ-uniform-b2",
            budget_bits: 2,
            b_min: 2,
            b_max: 2,
            policy: AllocPolicy::Uniform,
        },
        Variant {
            name: "VASQ-lin-b2",
            budget_bits: 2,
            b_min: 1,
            b_max: 6,
            policy: AllocPolicy::LinearVariance,
        },
        Variant {
            name: "VASQ-log-b2",
            budget_bits: 2,
            b_min: 1,
            b_max: 6,
            policy: AllocPolicy::LogVariance,
        },
        Variant {
            name: "SQ-uniform-b4",
            budget_bits: 4,
            b_min: 4,
            b_max: 4,
            policy: AllocPolicy::Uniform,
        },
        Variant {
            name: "VASQ-lin-b4",
            budget_bits: 4,
            b_min: 1,
            b_max: 8,
            policy: AllocPolicy::LinearVariance,
        },
        Variant {
            name: "VASQ-log-b4",
            budget_bits: 4,
            b_min: 1,
            b_max: 8,
            policy: AllocPolicy::LogVariance,
        },
        Variant {
            name: "SQ-uniform-b6",
            budget_bits: 6,
            b_min: 6,
            b_max: 6,
            policy: AllocPolicy::Uniform,
        },
        Variant {
            name: "VASQ-log-b6",
            budget_bits: 6,
            b_min: 2,
            b_max: 10,
            policy: AllocPolicy::LogVariance,
        },
        Variant {
            name: "SQ-uniform-b8",
            budget_bits: 8,
            b_min: 8,
            b_max: 8,
            policy: AllocPolicy::Uniform,
        },
    ];

    println!("{:<20} {:>8} {:>10} {:>12} {:>12} {:>12}",
             "variant", "bits/d", "recall@10", "us/query", "bytes_total", "KB");
    println!("{}", "-".repeat(78));

    let mut passed_accept = false;
    for v in &variants {
        let (recall, per_q_us, bytes_total, avg_bits) =
            run_variant(v, dim, &ds.train, &ds.base, &ds.queries, &gt, k);
        println!(
            "{:<20} {:>8.3} {:>10.4} {:>12.2} {:>12} {:>12.1}",
            v.name, avg_bits, recall, per_q_us, bytes_total, bytes_total as f64 / 1024.0
        );
        if v.name == "VASQ-log-b2" && recall >= 0.30 {
            passed_accept = true;
        }
    }

    println!();
    println!("# Acceptance test: VASQ-log-b2 recall@10 >= 0.30: {}",
             if passed_accept { "PASS" } else { "FAIL" });
}

fn min_f32(xs: &[f32]) -> f32 {
    xs.iter().cloned().fold(f32::INFINITY, f32::min)
}
fn max_f32(xs: &[f32]) -> f32 {
    xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}
fn median_f32(xs: &[f32]) -> f32 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
