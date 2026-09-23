//! Benchmark binary: compares ScalarPqAdc, Lut16ScalarFastscan,
//! Lut16BlockedFastscan on a synthetic Gaussian corpus. Emits real numbers
//! consumed by the accompanying research doc / gist.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_pq_fastscan_lut16::*;
use std::time::Instant;

fn make_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Isotropic zero-mean Gaussian via CLT approximation (sum of 12 uniforms).
    // This is the classical PQ benchmark shape -- no strong cluster modes,
    // so PQ codebook capacity translates directly into recall.
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d)
        .map(|_| {
            let s: f32 = (0..12).map(|_| rng.gen_range(0.0f32..1.0)).sum();
            s - 6.0
        })
        .collect()
}

fn bench<T: PqAdc>(name: &str, idx: &T, queries: &[f32], nq: usize, d: usize,
                   truth_top10: &[Vec<u32>], nprobe: usize) -> (f32, f64) {
    // "recall@nprobe of ground-truth top-10": fraction of the 10 exact NN
    // that appear in the PQ's top-`nprobe` shortlist. This is the standard
    // ANN metric used by Faiss / SCANN / RaBitQ.
    for q in 0..3 {
        let _ = idx.search(&queries[q * d..(q + 1) * d], nprobe);
    }
    let t0 = Instant::now();
    let mut total_recall = 0.0f32;
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let out: Vec<u32> = idx.search(q, nprobe).into_iter().map(|(i, _)| i).collect();
        total_recall += recall_at_k(&truth_top10[qi], &out);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let ms_per_q = elapsed * 1000.0 / nq as f64;
    let recall = total_recall / nq as f32;
    println!(
        "  {name:32}  R@{nprobe} of GT@10 = {recall:.3}   {ms_per_q:.3} ms/query   codes = {} KiB",
        idx.code_bytes() / 1024
    );
    (recall, ms_per_q)
}

fn main() {
    println!("=== ruvector-pq-fastscan-lut16 benchmark ===");
    let d = 64;
    let n = 50_000;
    let nq = 200;
    let gt_k = 10;
    let m = 8; // 64 / 8 = 8 dims per sub-quantizer
    println!("d={d} n={n} nq={nq} gt_k={gt_k} m={m}\n");

    println!("Building corpus + ground truth...");
    let data = make_data(n, d, 1234);
    let queries = make_data(nq, d, 5678);
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = (0..nq)
        .map(|qi| exact_topk(&data, n, d, &queries[qi * d..(qi + 1) * d], gt_k))
        .collect();
    println!("  ground truth in {:.2}s", t0.elapsed().as_secs_f64());

    println!("\nTraining ScalarPqAdc (ks=256, m={m})...");
    let t0 = Instant::now();
    let scalar = ScalarPqAdc::build(&data, n, d, m, 256, 20, 42).unwrap();
    println!("  trained in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Training Lut16ScalarFastscan (ks=16, m={m})...");
    let t0 = Instant::now();
    let fs_scalar = Lut16ScalarFastscan::build(&data, n, d, m, 20, 42).unwrap();
    println!("  trained in {:.2}s", t0.elapsed().as_secs_f64());

    println!("Training Lut16BlockedFastscan (ks=16, m={m}, BLOCK=32)...");
    let t0 = Instant::now();
    let fs_block = Lut16BlockedFastscan::build(&data, n, d, m, 20, 42).unwrap();
    println!("  trained in {:.2}s", t0.elapsed().as_secs_f64());

    println!("\n--- Search performance & recall ---");
    let nprobe = 100;
    let (r1, t1) = bench("ScalarPqAdc (ks=256, 8-bit)", &scalar, &queries, nq, d, &truth, nprobe);
    let (r2, t2) = bench("Lut16ScalarFastscan (ks=16)", &fs_scalar, &queries, nq, d, &truth, nprobe);
    let (r3, t3) = bench("Lut16BlockedFastscan (ks=16, block=32)", &fs_block, &queries, nq, d, &truth, nprobe);

    println!("\n--- Storage & speed relative to baseline ---");
    println!("  scan speedup: scalar-fs {:.2}x   blocked-fs {:.2}x", t1 / t2, t1 / t3);
    let scalar_bytes = scalar.code_bytes();
    let fs_scalar_bytes = fs_scalar.code_bytes();
    let fs_block_bytes = fs_block.code_bytes();
    println!(
        "  code size:   scalar {} KiB  fs-scalar {} KiB ({:.2}x)  fs-blocked {} KiB ({:.2}x)",
        scalar_bytes / 1024,
        fs_scalar_bytes / 1024,
        scalar_bytes as f32 / fs_scalar_bytes as f32,
        fs_block_bytes / 1024,
        scalar_bytes as f32 / fs_block_bytes as f32
    );
    println!("  R@{nprobe} of GT@{gt_k}:   scalar {r1:.3}  fs-scalar {r2:.3}  fs-blocked {r3:.3}");
}
