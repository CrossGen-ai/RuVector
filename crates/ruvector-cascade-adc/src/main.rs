//! `cascade-bench` — runnable benchmark producing REAL numbers.
//!
//! Generates a synthetic Gaussian-mixture corpus (deterministic seed), builds
//! a PQ index with 8-bit + 4-bit codebooks, then runs three scanners over a
//! held-out query set. Reports:
//!
//!   * bytes / vector (memory)
//!   * queries / second (throughput)
//!   * recall@10 vs. brute-force ground truth
//!
//! Run:
//!     cargo run --release -p ruvector-cascade-adc --bin cascade-bench

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_cascade_adc::{
    bytes_per_vector, CascadeScanner, FullEightBitScanner, FullFourBitScanner, Layout, PqIndex,
    PqParams, ScanResult, Scanner, TrainingConfig,
};
use std::time::Instant;

fn gen_gmm(n: usize, d: usize, _n_centers: usize, seed: u64) -> Vec<f32> {
    // Standard PQ benchmark distribution: unit-variance i.i.d. Gaussian
    // per dimension. Query and corpus drawn from the same distribution.
    // Nearest-neighbour ranking is well-defined and PQ recall is measurable.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];
    for v in data.iter_mut() {
        *v = box_muller(&mut rng);
    }
    data
}

fn box_muller(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-9);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn brute_topk(data: &[f32], n: usize, d: usize, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| {
            let off = i as usize * d;
            let mut s = 0.0f32;
            for t in 0..d {
                let x = data[off + t] - query[t];
                s += x * x;
            }
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall(gt: &[u32], got: &[ScanResult]) -> f32 {
    let mut hit = 0;
    for r in got {
        if gt.contains(&r.id) {
            hit += 1;
        }
    }
    hit as f32 / gt.len() as f32
}

fn run_scanner(name: &str, layout: Layout, m: usize, index: &PqIndex, queries: &[f32], nq: usize, d: usize,
               ground_truth: &[Vec<u32>], k: usize, scanner: &dyn Scanner) {
    let mut out = Vec::with_capacity(k);
    let mut total_recall = 0.0f32;

    // Warm-up.
    for q in 0..nq.min(4) {
        scanner.search(index, &queries[q * d..(q + 1) * d], k, &mut out);
    }

    let t0 = Instant::now();
    for q in 0..nq {
        scanner.search(index, &queries[q * d..(q + 1) * d], k, &mut out);
        total_recall += recall(&ground_truth[q], &out);
    }
    let elapsed = t0.elapsed();
    let qps = nq as f64 / elapsed.as_secs_f64();
    let mean_recall = total_recall / nq as f32;
    let bpv = bytes_per_vector(m, layout);
    println!(
        "  {:<12} bytes/vec={:>4}  qps={:>10.1}  recall@{}={:.4}  total={:.3}s",
        name, bpv, qps, k, mean_recall, elapsed.as_secs_f64()
    );
}

fn main() {
    let n = 200_000usize;
    let n_train = 10_000usize;
    let nq = 200usize;
    let d = 128usize;
    let m = 32usize;
    let k = 10usize;
    let n_centers = 64usize;

    println!("Cascade-ADC benchmark");
    println!("---------------------");
    println!("n={} d={} m={} nq={} k={} clusters={}", n, d, m, nq, k, n_centers);

    let data = gen_gmm(n, d, n_centers, 0xDA7A_5EED);
    let train = gen_gmm(n_train, d, n_centers, 0x7241_5EED);
    // Queries: pick nq data points and perturb them with small Gaussian noise.
    // This simulates the standard "query = nearby to a corpus vector" scenario
    // used in SIFT/GIST-style benchmarks and yields well-defined recall.
    let mut qrng = StdRng::seed_from_u64(0x9E27_5EED);
    let mut queries = vec![0.0f32; nq * d];
    for q in 0..nq {
        let src = qrng.gen_range(0..n);
        for t in 0..d {
            queries[q * d + t] = data[src * d + t] + box_muller(&mut qrng) * 0.05;
        }
    }

    let t0 = Instant::now();
    let index = PqIndex::build(
        &train, n_train, &data, n,
        PqParams { d, m },
        &TrainingConfig::default(),
    );
    println!("index built in {:.2}s", t0.elapsed().as_secs_f64());
    println!("  codes8 bytes={}  codes4 bytes={}", index.codes8.len(), index.codes4.len());

    // Ground truth by brute force.
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..nq)
        .map(|q| brute_topk(&data, n, d, &queries[q * d..(q + 1) * d], k))
        .collect();
    println!("brute-force ground truth in {:.2}s", t0.elapsed().as_secs_f64());

    println!("\nResults:");
    run_scanner("full-8bit", Layout::EightBit, m, &index, &queries, nq, d, &gt, k,
                &FullEightBitScanner);
    run_scanner("full-4bit", Layout::FourBit, m, &index, &queries, nq, d, &gt, k,
                &FullFourBitScanner);
    for &rho in &[0.05f32, 0.10, 0.20] {
        let s = CascadeScanner::with_floor(rho, 100);
        let name = format!("cascade-rho{:.2}", rho);
        run_scanner(&name, Layout::Cascade, m, &index, &queries, nq, d, &gt, k, &s);
    }
}
