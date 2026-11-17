//! Run SC-PQ and PQ on correlated synthetic data and print recall@10 and
//! scan throughput for both. Deterministic: use `cargo run --release
//! --example demo` and you will get the numbers quoted in the ADR.

use ruvector_subspace_conditional_pq::{
    data::gen_correlated, exact_top_k, recall_at_k, top_k, Pq, Quantizer, ScPq,
};
use std::time::Instant;

fn run(name: &str, q: &dyn Quantizer, data: &[f32], queries: &[f32], d: usize, k_top: usize) {
    let t0 = Instant::now();
    let codes = q.encode(data);
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let n_queries = queries.len() / d;
    let mut recall_sum = 0f32;
    let mut lut_us = 0f64;
    let mut scan_us = 0f64;
    for qi in 0..n_queries {
        let query = &queries[qi * d..(qi + 1) * d];
        let exact = exact_top_k(query, data, d, k_top);

        let t = Instant::now();
        let lut = q.build_lut(query);
        lut_us += t.elapsed().as_secs_f64() * 1_000_000.0;

        let t = Instant::now();
        let scores = q.score(&lut, &codes);
        scan_us += t.elapsed().as_secs_f64() * 1_000_000.0;

        let approx = top_k(&scores, k_top);
        recall_sum += recall_at_k(&approx, &exact);
    }
    let n = data.len() / d;
    let scan_us_mean = scan_us / n_queries as f64;
    let scan_mvec_per_s = n as f64 / scan_us_mean;
    println!(
        "{name:>8} | recall@{k_top}={:.3} | enc={:.1} ms | lut={:.1} us | scan={:.1} us ({:.2} Mvec/s)",
        recall_sum / n_queries as f32,
        enc_ms,
        lut_us / n_queries as f64,
        scan_us_mean,
        scan_mvec_per_s
    );
}

fn main() {
    let n = 20_000;
    let d = 64;
    let latent = 8; // strong correlation: 64 output dims from 8 latent factors
    let noise = 0.1;
    let m = 8;
    let k = 16;
    let iters = 10;
    let k_top = 10;

    let data = gen_correlated(n, d, latent, noise, 0xC0FFEE);
    let queries = gen_correlated(50, d, latent, noise, 0xBEEF);
    println!("N={n} D={d} latent={latent} noise={noise} M={m} K={k} iters={iters}");

    let pq = Pq::train(&data, d, m, k, iters, 1);
    let scpq = ScPq::train(&data, d, m, k, iters, 1);

    run("PQ", &pq, &data, &queries, d, k_top);
    run("SC-PQ", &scpq, &data, &queries, d, k_top);
}
