//! Benchmark binary: measures MSE, recall@10, encode throughput, and query
//! throughput for PlainPQ vs AnisotropicPQ vs AnisotropicPQR on a synthetic
//! mixture-of-Gaussians dataset.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release -p ruvector-anisotropic-pq --bin anisotropic-pq-bench
//! ```
//!
//! Numbers reported here are the ones cited in the research doc and gist —
//! CI/user should re-run to get hardware-specific figures.

use ruvector_anisotropic_pq::{
    gen_gauss, sq_l2, AnisotropicPQ, AnisotropicPQR, Code, PlainPQ, PqParams, Quantizer,
};
use std::time::Instant;

fn main() {
    // Modest sizes so `cargo run --release` finishes in a few seconds on CI.
    let dim = 64;
    let m = 8;
    let k = 256;
    let n = 20_000;
    let n_train = 8_000;
    let n_query = 200;
    let iters = 15;
    let seed = 2026_08_20;

    println!("== ruvector-anisotropic-pq benchmark ==");
    println!(
        "dim={dim} m={m} k={k} n={n} train={n_train} queries={n_query} iters={iters} seed={seed}"
    );
    println!("bytes/code = {m}  (compression = {}x vs f32)", dim * 4 / m);

    let data = gen_gauss(n, dim, 16, seed);
    let train: Vec<Vec<f32>> = data.iter().take(n_train).cloned().collect();
    let queries = gen_gauss(n_query, dim, 16, seed ^ 0xDEAD_BEEF);

    let params = PqParams {
        dim,
        m,
        k,
        iters,
        seed,
    };

    // --- Train all three variants and time it.
    let t = Instant::now();
    let plain = PlainPQ::train(&train, params).expect("plain train");
    let t_plain_train = t.elapsed();

    let t = Instant::now();
    let aniso = AnisotropicPQ::train(&train, params, 4.0).expect("aniso train");
    let t_aniso_train = t.elapsed();

    let t = Instant::now();
    let aniso_r = AnisotropicPQR::train(&train, params, 4.0).expect("aniso+rot train");
    let t_rot_train = t.elapsed();

    println!();
    println!("-- training time --");
    println!("plain      : {:?}", t_plain_train);
    println!("anisotropic: {:?}", t_aniso_train);
    println!("aniso+rot  : {:?}", t_rot_train);

    // --- Encode dataset.
    let (codes_p, enc_p) = bench_encode(&plain, &data);
    let (codes_a, enc_a) = bench_encode(&aniso, &data);
    let (codes_r, enc_r) = bench_encode(&aniso_r, &data);

    println!();
    println!("-- encode throughput (vectors/sec) --");
    println!("plain      : {:>10.0}", n as f64 / enc_p);
    println!("anisotropic: {:>10.0}", n as f64 / enc_a);
    println!("aniso+rot  : {:>10.0}   (extra rotation apply per vector)", n as f64 / enc_r);

    // --- MSE.
    let mse_p = mean_recon_mse(&plain, &data, &codes_p);
    let mse_a = mean_recon_mse(&aniso, &data, &codes_a);
    let mse_r = mean_recon_mse(&aniso_r, &data, &codes_r);
    println!();
    println!("-- reconstruction MSE (lower is better) --");
    println!("plain      : {mse_p:.5}");
    println!("anisotropic: {mse_a:.5}   ({:+.2}% vs plain)", pct(mse_a, mse_p));
    println!("aniso+rot  : {mse_r:.5}   ({:+.2}% vs plain)", pct(mse_r, mse_p));

    // --- Recall@10 vs exact top-10.
    let (recall_p, qps_p) = bench_recall(&plain, &data, &queries, &codes_p, 10);
    let (recall_a, qps_a) = bench_recall(&aniso, &data, &queries, &codes_a, 10);
    let (recall_r, qps_r) = bench_recall(&aniso_r, &data, &queries, &codes_r, 10);

    println!();
    println!("-- recall@10 (higher is better) & query throughput --");
    println!("plain      : recall={recall_p:.4}  qps={qps_p:>8.0}");
    println!(
        "anisotropic: recall={recall_a:.4}  qps={qps_a:>8.0}  ({:+.2}pp)",
        (recall_a - recall_p) * 100.0
    );
    println!(
        "aniso+rot  : recall={recall_r:.4}  qps={qps_r:>8.0}  ({:+.2}pp)",
        (recall_r - recall_p) * 100.0
    );

    // --- Memory footprint.
    let bytes_per = plain.bytes_per_code();
    let total_kb = (bytes_per * n) as f64 / 1024.0;
    println!();
    println!("-- storage --");
    println!("per-vector: {bytes_per} bytes   dataset: {total_kb:.1} KB");
    println!(
        "f32 baseline: {:.1} KB   compression: {:.1}x",
        (dim * 4 * n) as f64 / 1024.0,
        (dim * 4) as f64 / bytes_per as f64
    );

    println!();
    println!("== done ==");
}

fn bench_encode<Q: Quantizer>(q: &Q, data: &[Vec<f32>]) -> (Vec<Code>, f64) {
    let t = Instant::now();
    let codes: Vec<Code> = data.iter().map(|v| q.encode(v).unwrap()).collect();
    (codes, t.elapsed().as_secs_f64())
}

fn mean_recon_mse<Q: Quantizer>(q: &Q, data: &[Vec<f32>], codes: &[Code]) -> f32 {
    let mut acc = 0.0f32;
    for (v, c) in data.iter().zip(codes.iter()) {
        let r = q.reconstruct(c);
        acc += sq_l2(v, &r) / v.len() as f32;
    }
    acc / data.len() as f32
}

fn bench_recall<Q: Quantizer>(
    q: &Q,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    codes: &[Code],
    k: usize,
) -> (f32, f64) {
    let qcodes: Vec<Code> = queries.iter().map(|v| q.encode(v).unwrap()).collect();
    let t = Instant::now();
    let mut hit = 0usize;
    let mut total = 0usize;
    for (qi, qv) in queries.iter().enumerate() {
        let mut exact: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, sq_l2(qv, v)))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: std::collections::HashSet<usize> =
            exact.into_iter().take(k).map(|(i, _)| i).collect();
        let mut approx: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, q.sdc(&qcodes[qi], c)))
            .collect();
        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (i, _) in approx.into_iter().take(k) {
            if gt.contains(&i) {
                hit += 1;
            }
        }
        total += k;
    }
    let secs = t.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / secs;
    (hit as f32 / total as f32, qps)
}

fn pct(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        0.0
    } else {
        (a - b) / b * 100.0
    }
}
