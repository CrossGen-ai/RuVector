//! Standalone benchmark binary (not criterion) so `cargo bench` runs real
//! numbers without extra deps. Prints a comparison table for three regimes:
//!   - correlated / independent / high-correlation
//! Each line is one measurement run captured verbatim in the research doc.

use ruvector_subspace_conditional_pq::{
    data::{gen_correlated, gen_independent},
    exact_top_k, recall_at_k, top_k, Pq, Quantizer, ScPq,
};
use std::time::Instant;

struct Row {
    label: String,
    recall: f32,
    lut_us: f64,
    scan_us: f64,
    mvec_s: f64,
    code_bytes_per_vec: usize,
}

fn eval(name: &str, q: &dyn Quantizer, data: &[f32], queries: &[f32], d: usize) -> Row {
    let codes = q.encode(data);
    let n = data.len() / d;
    let nq = queries.len() / d;
    let k_top = 10;
    let mut recall_sum = 0f32;
    let mut lut_us = 0f64;
    let mut scan_us = 0f64;
    for qi in 0..nq {
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
    let scan_us_mean = scan_us / nq as f64;
    Row {
        label: name.to_string(),
        recall: recall_sum / nq as f32,
        lut_us: lut_us / nq as f64,
        scan_us: scan_us_mean,
        mvec_s: n as f64 / scan_us_mean,
        code_bytes_per_vec: codes.len() / n,
    }
}

fn scenario(name: &str, data: &[f32], queries: &[f32], d: usize, m: usize, k: usize) {
    let iters = 10;
    let pq = Pq::train(data, d, m, k, iters, 1);
    let scpq = ScPq::train(data, d, m, k, iters, 1);
    let rows = [
        eval("PQ", &pq, data, queries, d),
        eval("SC-PQ", &scpq, data, queries, d),
    ];
    println!("\n== {name} (N={}, D={d}, M={m}, K={k}) ==", data.len() / d);
    println!(
        "{:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>6}",
        "scheme", "recall@10", "lut_us", "scan_us", "Mvec/s", "bytes"
    );
    for r in rows {
        println!(
            "{:>8} | {:>9.3} | {:>10.1} | {:>10.1} | {:>10.2} | {:>6}",
            r.label, r.recall, r.lut_us, r.scan_us, r.mvec_s, r.code_bytes_per_vec
        );
    }
}

fn main() {
    let n = 20_000;
    let d = 64;
    // High correlation: 4 latents into 64 dims.
    let high = gen_correlated(n, d, 4, 0.05, 0xC0FFEE);
    let highq = gen_correlated(50, d, 4, 0.05, 0xBEEF);
    scenario("HIGH correlation (latent=4,noise=0.05)", &high, &highq, d, 8, 16);

    // Medium.
    let med = gen_correlated(n, d, 8, 0.1, 0xC0FFEE);
    let medq = gen_correlated(50, d, 8, 0.1, 0xBEEF);
    scenario("MED correlation (latent=8,noise=0.1)", &med, &medq, d, 8, 16);

    // Independent: SC-PQ should roughly match PQ.
    let ind = gen_independent(n, d, 0xDEADBEEF);
    let indq = gen_independent(50, d, 0x1337);
    scenario("INDEPENDENT (isotropic Gaussian)", &ind, &indq, d, 8, 16);

    // Smaller M (longer subvectors): correlation harder to exploit.
    scenario("MED, M=4", &med, &medq, d, 4, 16);
}
