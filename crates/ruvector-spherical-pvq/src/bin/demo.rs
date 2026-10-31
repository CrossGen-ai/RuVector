//! End-to-end PVQ demo + measurement harness.
//!
//! Builds a synthetic unit-norm dataset, encodes it under three PVQ budgets,
//! and reports: MSE, bytes/vector, encode speed, scan latency, recall@10
//! against float32 flat ground truth.
//!
//! Reproducibility: seeded ChaCha RNG, single-threaded, release profile.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use ruvector_spherical_pvq::{
    flat_cosine_topk, l2_normalize, recall_at, reconstruction_mse, PvqCodec, PvqIndex,
};
use std::time::Instant;

fn random_unit(d: usize, rng: &mut ChaCha12Rng) -> Vec<f32> {
    // Mixed anisotropy: 70% isotropic Gaussian, 30% skewed to a random axis
    // so a few dims dominate — exercises PVQ's L1-sphere coverage.
    let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
    if rng.gen::<f32>() < 0.3 {
        let axis = rng.gen_range(0..d);
        v[axis] += 2.0 * (rng.gen::<f32>() - 0.5) * 3.0;
    }
    l2_normalize(&mut v);
    v
}

fn flat_search(vectors: &[Vec<f32>], queries: &[Vec<f32>], topk: usize) -> (f64, Vec<Vec<usize>>) {
    let t0 = Instant::now();
    let gts: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| flat_cosine_topk(vectors, q, topk))
        .collect();
    let elapsed = t0.elapsed().as_secs_f64();
    (elapsed, gts)
}

fn run_pvq(
    label: &str,
    d: usize,
    k: u32,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    gts: &[Vec<usize>],
    topk: usize,
) {
    let codec = PvqCodec::new(d, k);

    // Encode phase
    let t0 = Instant::now();
    let index = PvqIndex::build(codec.clone(), vectors);
    let encode_s = t0.elapsed().as_secs_f64();

    // MSE over a sample
    let mut mse_sum = 0.0;
    let sample = 500.min(vectors.len());
    for v in vectors.iter().take(sample) {
        mse_sum += reconstruction_mse(&codec, v);
    }
    let mse = mse_sum / sample as f32;

    // Scan phase
    let t0 = Instant::now();
    let mut recall_sum = 0.0;
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let hits = index.search(q, topk);
        recall_sum += recall_at(gt, &hits);
    }
    let scan_s = t0.elapsed().as_secs_f64();
    let recall = recall_sum / queries.len() as f32;

    let per_q_us = scan_s / queries.len() as f64 * 1e6;
    let bytes_per_vec = codec.bytes_per_vector();
    let total_bytes_mb = index.bytes_total() as f64 / (1024.0 * 1024.0);

    println!(
        "[{label}]  K={k:>3}  bytes/vec={bytes_per_vec:>4}  index={total_bytes_mb:>5.2}MiB  \
         MSE={mse:.4e}  encode={encode_s:>5.2}s  scan={per_q_us:>6.1}µs/q  recall@{topk}={recall:.3}"
    );
}

fn main() {
    let d = 128;
    let n = 20_000;
    let n_queries = 200;
    let topk = 10;
    let seed = 0xC0FFEE_u64;

    println!("ruvector-spherical-pvq demo");
    println!("  dim        : {d}");
    println!("  n_vectors  : {n}");
    println!("  n_queries  : {n_queries}");
    println!("  top-k      : {topk}");
    println!("  seed       : 0x{seed:x}");
    println!();

    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_unit(d, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| random_unit(d, &mut rng)).collect();

    // Baseline flat float32.
    let (flat_s, gts) = flat_search(&vectors, &queries, topk);
    let per_q_us = flat_s / queries.len() as f64 * 1e6;
    let flat_bytes_per_vec = d * 4;
    let flat_total_mb = (n * d * 4) as f64 / (1024.0 * 1024.0);
    println!(
        "[flat-f32] K= -   bytes/vec={flat_bytes_per_vec:>4}  index={flat_total_mb:>5.2}MiB  \
         MSE=0.00e0    encode= 0.00s  scan={per_q_us:>6.1}µs/q  recall@{topk}=1.000"
    );

    // Three PVQ budgets.
    run_pvq("pvq-coarse", d, 16, &vectors, &queries, &gts, topk);
    run_pvq("pvq-med   ", d, 32, &vectors, &queries, &gts, topk);
    run_pvq("pvq-fine  ", d, 64, &vectors, &queries, &gts, topk);
}
