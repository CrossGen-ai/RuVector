//! Demo binary: measure recall@10 and per-query latency for F32Flat,
//! SymRabitQ, and AsymQ2D1 on a synthetic Gaussian corpus.

use ruvector_asymmetric_rabitq_q2d1::{
    gen_corpus, recall_at_k, Backend, TriIndex,
};
use std::time::Instant;

fn bench(label: &str, idx: &TriIndex, queries: &[Vec<f32>], truth: &[Vec<(usize, f32)>], backend: Backend, tau: f32, k: usize) {
    let t0 = Instant::now();
    let mut recall_sum = 0.0f32;
    let mut cnt = 0u64;
    for (i, q) in queries.iter().enumerate() {
        let pred = idx.topk(q, k, backend, tau);
        recall_sum += recall_at_k(&pred, &truth[i], k);
        cnt += 1;
    }
    let total_us = t0.elapsed().as_micros() as f64;
    let per_us = total_us / cnt as f64;
    let recall = recall_sum / cnt as f32;
    println!(
        "{label:>14}  recall@{k} = {recall:.4}   {per_us:>7.1} us/query   total {total_us:.0} us"
    );
}

fn main() {
    let n = 50_000;
    let dim = 128;
    let nq = 500;
    let k = 10;

    println!("# asymmetric-rabitq-q2d1 demo");
    println!("# N = {n}, dim = {dim}, nq = {nq}, k = {k}");

    let corpus = gen_corpus(n, dim, 42);
    let queries = gen_corpus(nq, dim, 99);

    let t0 = Instant::now();
    let idx = TriIndex::build(&corpus, 11);
    println!("# build: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Ground truth is F32Flat on the rotated vectors (correct because
    // rotation is isometric up to a global scale).
    let truth: Vec<Vec<(usize, f32)>> = queries
        .iter()
        .map(|q| idx.topk(q, k, Backend::F32Flat, 0.0))
        .collect();

    bench("F32Flat",    &idx, &queries, &truth, Backend::F32Flat,   0.0, k);
    bench("SymRabitQ",  &idx, &queries, &truth, Backend::SymRabitQ, 0.0, k);
    for tau in [0.25f32, 0.5, 0.75] {
        let label = format!("AsymQ2D1(τ={tau})");
        bench(&label, &idx, &queries, &truth, Backend::AsymQ2D1, tau, k);
    }

    // Storage math
    let bits_per_vec = dim; // 1 bit/dim
    let bytes_per_vec = (bits_per_vec + 7) / 8;
    println!(
        "# storage: {} bytes/vec @ 1 bit/dim  (vs {} bytes/vec float)",
        bytes_per_vec,
        dim * 4
    );
}
