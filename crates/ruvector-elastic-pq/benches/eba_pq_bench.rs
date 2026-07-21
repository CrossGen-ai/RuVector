//! Standalone benchmark binary (Cargo bench with `harness = false`).
//!
//! Trains three PQ variants on a shared synthetic anisotropic corpus and
//! prints one CSV line per variant plus a human table. The research doc
//! captures the output verbatim so numbers travel with the artifact.
//!
//! Run: `cargo bench -p ruvector-elastic-pq`
//! or:   `cargo run --release -p ruvector-elastic-pq --bench eba_pq_bench`
//! Output is stable across runs (fixed seeds).

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_elastic_pq::{AdcSearcher, Allocator, ElasticPqBuilder};

fn make_anisotropic(n: usize, dim: usize, m: usize, seed: u64) -> Vec<f32> {
    let sub_dim = dim / m;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0f32; n * dim];
    let scales: Vec<f32> = (0..m).map(|i| 1.0 / (1.0 + i as f32).sqrt()).collect();
    for i in 0..n {
        for s in 0..m {
            let sigma = scales[s];
            for d in 0..sub_dim {
                let z: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                out[i * dim + s * sub_dim + d] = z * sigma;
            }
        }
    }
    out
}

fn truth_topk(query: &[f32], base: &[f32], n: usize, dim: usize, k: usize) -> Vec<u32> {
    let mut with_d: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let mut acc = 0f32;
            for d in 0..dim {
                let diff = query[d] - base[i * dim + d];
                acc += diff * diff;
            }
            (i as u32, acc)
        })
        .collect();
    with_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    with_d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at(truth: &[u32], approx: &[u32]) -> f32 {
    let mut hits = 0;
    for a in approx {
        if truth.contains(a) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

struct RunResult {
    name: String,
    bits_per_sub: Vec<u8>,
    total_bits: usize,
    train_distortion: f64,
    train_ms: u128,
    encode_ms: u128,
    search_ms_avg: f64,
    recall_at_10: f32,
    swaps: usize,
}

fn run_variant(
    label: &str,
    allocator: Allocator,
    base: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    queries: &[f32],
    n_queries: usize,
    truth: &[Vec<u32>],
    k: usize,
) -> RunResult {
    let t0 = Instant::now();
    let pq = ElasticPqBuilder::new(m)
        .allocator(allocator)
        .seed(0xC0FFEE)
        .train(base, n, dim)
        .expect("train");
    let train_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    let codes = pq.encode_batch(base, n);
    let encode_ms = t1.elapsed().as_millis();

    let searcher = AdcSearcher::new(&pq, &codes, n);
    let mut recall_sum = 0f32;
    let mut search_ns_total = 0u128;
    for q in 0..n_queries {
        let query = &queries[q * dim..(q + 1) * dim];
        let ts = Instant::now();
        let approx = searcher.topk(query, k).expect("topk");
        search_ns_total += ts.elapsed().as_nanos();
        let approx_ids: Vec<u32> = approx.iter().map(|r| r.index).collect();
        recall_sum += recall_at(&truth[q], &approx_ids);
    }
    let search_ms_avg = search_ns_total as f64 / 1_000_000.0 / n_queries as f64;

    RunResult {
        name: label.to_string(),
        bits_per_sub: pq.stats().bits.clone(),
        total_bits: pq.stats().total_bits,
        train_distortion: pq.stats().total_distortion,
        train_ms,
        encode_ms,
        search_ms_avg,
        recall_at_10: recall_sum / n_queries as f32,
        swaps: pq.stats().swaps,
    }
}

fn main() {
    let n = 5_000;
    let dim = 32;
    let m = 8;
    let k = 10;
    let n_queries = 200;

    println!("EBA-PQ benchmark (n={}, dim={}, m={}, k={})", n, dim, m, k);
    println!("Corpus: anisotropic synthetic (sigma_s = 1/sqrt(1+s))");
    println!();

    let base = make_anisotropic(n, dim, m, 0xDA7A);
    let queries = make_anisotropic(n_queries, dim, m, 0x9CE);

    let truth: Vec<Vec<u32>> = (0..n_queries)
        .map(|q| truth_topk(&queries[q * dim..(q + 1) * dim], &base, n, dim, k))
        .collect();

    let variants: Vec<(&str, Allocator)> = vec![
        ("Uniform-4bit", Allocator::Uniform { bits: 4 }),
        (
            "VarianceProp-32b",
            Allocator::VarianceProportional {
                total_bits: 32,
                min_bits: 2,
                max_bits: 6,
            },
        ),
        (
            "ElasticIter-32b",
            Allocator::DistortionIterative {
                start_bits: 4,
                min_bits: 2,
                max_bits: 6,
                max_swaps: 32,
            },
        ),
    ];

    let mut rows = Vec::new();
    for (label, alloc) in variants {
        let r = run_variant(label, alloc, &base, n, dim, m, &queries, n_queries, &truth, k);
        rows.push(r);
    }

    // Human table.
    println!(
        "{:<20} {:>10} {:>14} {:>10} {:>10} {:>12} {:>10} {:>6}",
        "variant", "total_bits", "distortion", "train_ms", "enc_ms", "search_us", "recall@10", "swaps"
    );
    for r in &rows {
        println!(
            "{:<20} {:>10} {:>14.4} {:>10} {:>10} {:>12.2} {:>10.3} {:>6}",
            r.name,
            r.total_bits,
            r.train_distortion,
            r.train_ms,
            r.encode_ms,
            r.search_ms_avg * 1000.0,
            r.recall_at_10,
            r.swaps
        );
    }

    // CSV footer.
    println!();
    println!("variant,total_bits,train_distortion,train_ms,encode_ms,search_us_avg,recall_at_10,swaps,bits_per_subspace");
    for r in &rows {
        let bits_str: Vec<String> = r.bits_per_sub.iter().map(|b| b.to_string()).collect();
        println!(
            "{},{},{:.6},{},{},{:.4},{:.4},{},{}",
            r.name,
            r.total_bits,
            r.train_distortion,
            r.train_ms,
            r.encode_ms,
            r.search_ms_avg * 1000.0,
            r.recall_at_10,
            r.swaps,
            bits_str.join(":")
        );
    }
}
