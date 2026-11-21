//! Benchmark for polysemous PQ vs plain ADC vs exact search.
//!
//! Synthesizes a mixture-of-Gaussians dataset (clustered, closer to real
//! embedding distributions than uniform noise), trains PQ, applies the
//! polysemous SA reordering, then measures:
//!   * build cost
//!   * query latency per variant
//!   * recall@10 vs exact L2 ground truth
//!   * survivor count after Hamming prefilter

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use ruvector_polysemous_pq_hamming_prefilter::*;
use std::time::Instant;

fn make_mog_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    // Clusters in a hypercube [-3, 3] with σ≈0.4 noise: clusters are
    // well-separated, each item has ~ n/n_clusters near-neighbours,
    // mimicking a realistic embedding manifold.
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut centers = vec![0f32; n_clusters * dim];
    for v in centers.iter_mut() {
        *v = (rng.gen::<f32>() - 0.5) * 6.0;
    }
    let mut out = vec![0f32; n * dim];
    for i in 0..n {
        let c = rng.gen_range(0..n_clusters);
        for d in 0..dim {
            // Approx Gaussian via sum of 4 uniforms (CLT, σ ≈ 0.4).
            let noise: f32 = (0..4).map(|_| rng.gen::<f32>() - 0.5).sum::<f32>() * 0.4;
            out[i * dim + d] = centers[c * dim + d] + noise;
        }
    }
    out
}

fn mean_time_ms<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    // Warmup.
    f();
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed / reps as f64
}

fn main() {
    // Parameters tuned so build+bench finishes in < 60s on a laptop.
    let n = 50_000;
    let dim = 64;
    let m = 8; // 8 subspaces of 8 dim each
    let ksub = 256; // 8-bit codes → 64-bit PQ codes
    let n_clusters = 2500; // ~ 20 vectors / cluster — top-10 is roughly 1 cluster
    let n_queries = 200;
    let top_k = 10;
    let km_iters = 20;
    let sa_iters = 80_000;

    println!("=== ruvector-polysemous-pq-hamming-prefilter benchmark ===");
    println!(
        "n={n}, dim={dim}, m={m}, ksub={ksub}, queries={n_queries}, top_k={top_k}, \
         km_iters={km_iters}, sa_iters={sa_iters}"
    );
    println!();

    print!("Generating MoG data... ");
    let t = Instant::now();
    let data = make_mog_data(n, dim, n_clusters, 1234);
    // Queries drawn from same distribution but a different seed so they are
    // in-distribution without being literal duplicates.
    let queries = make_mog_data(n_queries, dim, n_clusters, 2025);
    println!("done ({:.2}s)", t.elapsed().as_secs_f32());

    print!("Training PQ... ");
    let t = Instant::now();
    let mut pq = Pq::train(&data, dim, m, ksub, km_iters, 42).unwrap();
    let train_time = t.elapsed().as_secs_f32();
    println!("done ({train_time:.2}s)");

    print!("Fitting polysemous SA ({sa_iters} iters × {m} subspaces)... ");
    let t = Instant::now();
    pq.fit_polysemous(sa_iters, 99);
    let sa_time = t.elapsed().as_secs_f32();
    println!("done ({sa_time:.2}s)");

    print!("Encoding codes... ");
    let t = Instant::now();
    let codes = pack_codes(&pq, &data);
    let enc_time = t.elapsed().as_secs_f32();
    println!("done ({enc_time:.2}s)");
    let code_bytes = codes.len();
    let raw_bytes = data.len() * 4;
    println!(
        "  raw vectors: {:.2} MiB, PQ codes: {:.2} MiB  (compression {:.1}x)",
        raw_bytes as f64 / (1 << 20) as f64,
        code_bytes as f64 / (1 << 20) as f64,
        raw_bytes as f64 / code_bytes as f64
    );
    println!();

    // Build ground truth with exact search.
    print!("Building exact ground truth for {n_queries} queries... ");
    let t = Instant::now();
    let mut gts = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let query = &queries[q * dim..(q + 1) * dim];
        gts.push(exact_search(&data, dim, n, query, top_k));
    }
    let gt_time = t.elapsed().as_secs_f32();
    println!("done ({gt_time:.2}s)");

    // Variant 1: exact scan time (reference).
    let exact_ms = mean_time_ms(
        || {
            let q = &queries[0..dim];
            let _ = exact_search(&data, dim, n, q, top_k);
        },
        5,
    );

    // Variant 2: ADC only.
    let adc_ms = mean_time_ms(
        || {
            let q = &queries[0..dim];
            let _ = adc_search(&pq, &codes, n, q, top_k);
        },
        20,
    );
    let mut adc_recall_sum = 0f32;
    for q in 0..n_queries {
        let query = &queries[q * dim..(q + 1) * dim];
        let r = adc_search(&pq, &codes, n, query, top_k);
        adc_recall_sum += recall(&gts[q], &r, top_k);
    }
    let adc_recall = adc_recall_sum / n_queries as f32;

    // Variant 3: Polysemous Hamming prefilter at multiple thresholds + ADC rerank.
    let max_bits = (m * 8) as u32; // for ksub=256
    println!();
    println!("--- Polysemous prefilter sweep (ADC rerank on survivors) ---");
    println!(
        "{:<10} {:>10} {:>14} {:>14} {:>14}",
        "thresh", "recall@10", "survivors", "lat ms", "speedup vs ADC"
    );
    let sweep = [
        max_bits / 4,       // 16
        max_bits * 5 / 16,  // 20
        max_bits * 3 / 8,   // 24
        max_bits * 7 / 16,  // 28
        max_bits / 2,       // 32
        max_bits * 9 / 16,  // 36
        max_bits * 5 / 8,   // 40
        max_bits * 3 / 4,   // 48
        max_bits,           // 64 (= plain ADC)
    ];
    for thresh in sweep {
        let t_ms = mean_time_ms(
            || {
                let q = &queries[0..dim];
                let _ = polysemous_search(&pq, &codes, n, q, thresh, top_k);
            },
            20,
        );
        let mut rsum = 0f32;
        let mut surv_sum = 0usize;
        for q in 0..n_queries {
            let query = &queries[q * dim..(q + 1) * dim];
            let (r, s) = polysemous_search(&pq, &codes, n, query, thresh, top_k);
            rsum += recall(&gts[q], &r, top_k);
            surv_sum += s;
        }
        let rec = rsum / n_queries as f32;
        let surv = surv_sum / n_queries;
        println!(
            "{:<10} {:>10.3} {:>14} {:>14.3} {:>14.2}x",
            thresh,
            rec,
            surv,
            t_ms,
            adc_ms / t_ms
        );
    }

    println!();
    println!("--- Reference variants ---");
    println!("exact L2 scan   : lat {exact_ms:.3} ms/query");
    println!("ADC (plain PQ)  : lat {adc_ms:.3} ms/query, recall@10 {adc_recall:.3}");
    println!();
    println!("Summary:");
    println!("  train_time     = {train_time:.2}s");
    println!("  sa_time        = {sa_time:.2}s");
    println!("  encode_time    = {enc_time:.2}s");
    println!("  code size      = {:.2} MiB", code_bytes as f64 / (1 << 20) as f64);
}
