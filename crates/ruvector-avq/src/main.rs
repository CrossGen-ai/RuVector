//! Benchmark binary. Prints real Recall@10 / bytes-per-vector /
//! query-latency numbers across PqMse, AvqScoreAware, AvqNorm at
//! M ∈ {8, 16}.

use std::time::Instant;

use ruvector_avq::avq::AvqScoreAware;
use ruvector_avq::avq_norm::AvqNorm;
use ruvector_avq::data::synth_corpus;
use ruvector_avq::pq::PqMse;
use ruvector_avq::{
    brute_top_k, recall_at_k, top_k_scores, AnisotropicConfig, Quantizer, QuantizerConfig,
};

fn main() {
    let dim = 128;
    let n_base = 20_000;
    let n_queries = 200;
    let n_train = 20_000;
    let n_clusters = 32;
    let k = 10;
    let seed: u64 = 0xA5EE_D2026_u64;

    let corpus = synth_corpus(dim, n_base, n_queries, n_train, n_clusters, seed);
    println!(
        "corpus: dim={dim} n_base={n_base} n_queries={n_queries} \
         n_train={n_train} n_clusters={n_clusters} k={k}"
    );

    // Ground truth.
    let t0 = Instant::now();
    let truth: Vec<Vec<usize>> = corpus
        .queries
        .iter()
        .map(|q| brute_top_k(q, &corpus.base, k))
        .collect();
    let bt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("brute-force ground truth: {:.2} ms total\n", bt_ms);

    // Header.
    println!(
        "{:<10} {:>3} {:>4} {:>10} {:>12} {:>12} {:>12}",
        "variant", "M", "Ks", "recall@10", "encode_ms", "adc_us/q", "bytes/vec"
    );

    for &m in &[8usize, 16] {
        run(
            "PqMse",
            m,
            dim,
            &corpus.train,
            &corpus.base,
            &corpus.queries,
            &truth,
            k,
            None,
            seed,
        );
        run(
            "Avq",
            m,
            dim,
            &corpus.train,
            &corpus.base,
            &corpus.queries,
            &truth,
            k,
            Some(false),
            seed,
        );
        run(
            "AvqNorm",
            m,
            dim,
            &corpus.train,
            &corpus.base,
            &corpus.queries,
            &truth,
            k,
            Some(true),
            seed,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    label: &str,
    m: usize,
    dim: usize,
    train: &[Vec<f32>],
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    k: usize,
    avq_kind: Option<bool>, // None = PQ, Some(false) = Avq, Some(true) = AvqNorm
    seed: u64,
) {
    let ks = 256;
    let cfg = QuantizerConfig { m, ks, iters: 12, seed };
    let aniso = AnisotropicConfig { t: 0.2 };

    match avq_kind {
        None => {
            let mut q = PqMse::new(cfg, dim).unwrap();
            q.train(train).unwrap();
            let t = Instant::now();
            let codes = q.encode(base).unwrap();
            let enc_ms = t.elapsed().as_secs_f64() * 1000.0;
            let bytes = q.code_bytes() + q.side_bytes();
            report(label, m, ks, &q, queries, truth, k, &codes, enc_ms, bytes);
        }
        Some(false) => {
            let mut q = AvqScoreAware::new(cfg, aniso, dim).unwrap();
            q.train(train).unwrap();
            let t = Instant::now();
            let codes = q.encode(base).unwrap();
            let enc_ms = t.elapsed().as_secs_f64() * 1000.0;
            let bytes = q.code_bytes() + q.side_bytes();
            report(label, m, ks, &q, queries, truth, k, &codes, enc_ms, bytes);
        }
        Some(true) => {
            let mut q = AvqNorm::new(cfg, aniso, dim).unwrap();
            q.train(train).unwrap();
            let t = Instant::now();
            let codes = q.encode_with_norms(base).unwrap();
            let enc_ms = t.elapsed().as_secs_f64() * 1000.0;
            let bytes = q.code_bytes() + q.side_bytes();
            report(label, m, ks, &q, queries, truth, k, &codes, enc_ms, bytes);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    label: &str,
    m: usize,
    ks: usize,
    q: &dyn Quantizer,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    k: usize,
    codes: &[u8],
    enc_ms: f64,
    bytes: usize,
) {
    let t = Instant::now();
    let mut recall_sum = 0.0f32;
    for (qi, query) in queries.iter().enumerate() {
        let scores = q.adc(query, codes).unwrap();
        let preds = top_k_scores(&scores, k);
        recall_sum += recall_at_k(&preds, &truth[qi]);
    }
    let adc_us = t.elapsed().as_secs_f64() * 1_000_000.0 / queries.len() as f64;
    let recall = recall_sum / queries.len() as f32;
    println!(
        "{:<10} {:>3} {:>4} {:>10.4} {:>12.1} {:>12.1} {:>12}",
        label, m, ks, recall, enc_ms, adc_us, bytes
    );
}
