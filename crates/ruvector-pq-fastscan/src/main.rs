//! Demo and benchmark binary for PQ FastScan.
//!
//! Generates a synthetic Gaussian-mixture dataset, builds three indices
//! (flat / PQ8 / FastScan-4) and prints real throughput + recall@10.
//!
//! Usage:
//!   cargo run --release -p ruvector-pq-fastscan
//!   N=200000 D=128 M=16 Q=200 cargo run --release -p ruvector-pq-fastscan

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_pq_fastscan::{flat_l2_topk, FastScanIndex, Pq8Index, ProductQuantizer};
use std::time::Instant;

fn gen_dataset(n: usize, d: usize, intrinsic_dim: usize, seed: u64) -> Vec<f32> {
    // SIFT/GIST-like generator: low-rank Gaussian (sample a low-dim latent,
    // project up to `d`, add light per-dim noise). Realistic ANN data has
    // most of its variance concentrated in <50 directions; pure isotropic
    // Gaussian or cluster-mixture both produce pathological PQ behavior
    // (the former because there is no compressible structure, the latter
    // because centroids collapse onto cluster modes and lose intra-cluster
    // resolution). Low-rank Gaussian is the canonical synthetic.
    let mut rng = StdRng::seed_from_u64(seed);
    let proj_n = Normal::new(0f32, 1f32).unwrap();
    let proj: Vec<f32> = (0..intrinsic_dim * d).map(|_| proj_n.sample(&mut rng)).collect();
    let latent_n = Normal::new(0f32, 1f32).unwrap();
    let noise_n = Normal::new(0f32, 0.1f32).unwrap();
    let mut data = vec![0f32; n * d];
    let mut latent = vec![0f32; intrinsic_dim];
    for i in 0..n {
        for j in 0..intrinsic_dim { latent[j] = latent_n.sample(&mut rng); }
        for k in 0..d {
            let mut acc = 0f32;
            for j in 0..intrinsic_dim {
                acc += latent[j] * proj[j * d + k];
            }
            data[i * d + k] = acc + noise_n.sample(&mut rng);
        }
    }
    data
}

fn recall_at_k(approx: &[u32], truth: &[(u32, f32)], k: usize) -> f32 {
    let truth_set: std::collections::HashSet<u32> = truth.iter().take(k).map(|&(i, _)| i).collect();
    let hits = approx.iter().take(k).filter(|i| truth_set.contains(i)).count();
    hits as f32 / k as f32
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let n = env_usize("N", 100_000);
    let d = env_usize("D", 128);
    let m = env_usize("M", 16);
    let q_count = env_usize("Q", 100);
    let k = env_usize("K", 10);
    let train_n = env_usize("TRAIN", 20_000).min(n);

    assert!(d % m == 0, "D ({}) must be divisible by M ({})", d, m);

    println!("PQ FastScan — synthetic benchmark");
    println!("  n={} d={} m={} train={} queries={} k={}", n, d, m, train_n, q_count, k);
    println!("  subspace dim = {} | code rate = {} bytes/vec (pq8) | {} bytes/vec (fastscan4)",
             d / m, m, m / 2);

    let t0 = Instant::now();
    let intrinsic = env_usize("INTRINSIC", 16);
    let data = gen_dataset(n, d, intrinsic, 0xA5A5_A5A5);
    // Queries are perturbed samples of database points (held out from the
    // training set) — this matches the standard ANN benchmark setup. Drawing
    // queries from a totally disjoint distribution would test out-of-domain
    // generalization of the codebook, which is a separate (much harder)
    // question and obscures the kernel-quality result we want to measure.
    let queries: Vec<f32> = {
        let mut rng = StdRng::seed_from_u64(0x5A5A_5A5A);
        let noise = Normal::new(0f32, 0.1f32).unwrap();
        let mut q = vec![0f32; q_count * d];
        for i in 0..q_count {
            let src = rng.gen_range(train_n..n); // held out of training
            for j in 0..d {
                q[i * d + j] = data[src * d + j] + noise.sample(&mut rng);
            }
        }
        q
    };
    println!("  data gen: {:?}", t0.elapsed());

    let train = &data[..train_n * d];

    // ---- Train PQ8 (K=256) ----
    let t0 = Instant::now();
    let pq8 = ProductQuantizer::train(train, train_n, d, m, 256, 12, 1).unwrap();
    let t_train_pq8 = t0.elapsed();
    let t0 = Instant::now();
    let pq8_idx = Pq8Index::from_vectors(pq8, &data, n);
    let t_build_pq8 = t0.elapsed();

    // ---- Train FastScan (K=16) ----
    let t0 = Instant::now();
    let fs = FastScanIndex::from_vectors(train, train_n, &data, n, d, m, 12, 2).unwrap();
    let t_build_fs = t0.elapsed();

    println!("  pq8 train: {:?} | pq8 encode: {:?}", t_train_pq8, t_build_pq8);
    println!("  fastscan train+encode: {:?}", t_build_fs);
    println!("  pq8 storage: {} KB | fastscan storage: {} KB | flat: {} KB",
             pq8_idx.codes.len() / 1024,
             fs.packed_bytes() / 1024,
             (data.len() * 4) / 1024);

    // ---- Bench: flat (truth) ----
    let mut truth_all: Vec<Vec<(u32, f32)>> = Vec::with_capacity(q_count);
    let t0 = Instant::now();
    for q in 0..q_count {
        let qv = &queries[q * d..(q + 1) * d];
        truth_all.push(flat_l2_topk(&data, n, d, qv, k));
    }
    let t_flat = t0.elapsed();

    // ---- Bench: PQ8 scan ----
    let mut pq8_top: Vec<Vec<u32>> = Vec::with_capacity(q_count);
    let t0 = Instant::now();
    for q in 0..q_count {
        let qv = &queries[q * d..(q + 1) * d];
        let r = pq8_idx.search(qv, k);
        pq8_top.push(r.iter().map(|&(i, _)| i).collect());
    }
    let t_pq8 = t0.elapsed();

    // ---- Bench: FastScan (raw) ----
    let mut fs_top: Vec<Vec<u32>> = Vec::with_capacity(q_count);
    let t0 = Instant::now();
    for q in 0..q_count {
        let qv = &queries[q * d..(q + 1) * d];
        let lut = fs.build_lut(qv);
        let r = fs.search_u16(&lut, k);
        fs_top.push(r.iter().map(|&(i, _)| i).collect());
    }
    let t_fs = t0.elapsed();

    // ---- Bench: FastScan + rerank ----
    let rerank_candidates = env_usize("RERANK", 10 * k);
    let mut fsr_top: Vec<Vec<u32>> = Vec::with_capacity(q_count);
    let t0 = Instant::now();
    for q in 0..q_count {
        let qv = &queries[q * d..(q + 1) * d];
        let lut = fs.build_lut(qv);
        let r = fs.search_rerank(&lut, qv, &data, d, rerank_candidates, k);
        fsr_top.push(r.iter().map(|&(i, _)| i).collect());
    }
    let t_fsr = t0.elapsed();

    // Diagnostic for query 0.
    {
        let q0 = &queries[0..d];
        let truth = &truth_all[0];
        let pq8r = pq8_idx.search(q0, 10);
        println!("  diag q0 truth top3: {:?}", truth.iter().take(3).collect::<Vec<_>>());
        println!("  diag q0 pq8   top3: {:?}", pq8r.iter().take(3).collect::<Vec<_>>());
        // Compute PQ8 approx distance for the true #1 vs PQ8's #1.
        let lut = pq8_idx.pq.build_lut_f32(q0);
        let approx = |i: u32| -> f32 {
            let m_ = pq8_idx.pq.m;
            let kc_ = pq8_idx.pq.k;
            let row = &pq8_idx.codes[(i as usize) * m_..(i as usize + 1) * m_];
            (0..m_).map(|s| lut[s * kc_ + row[s] as usize]).sum()
        };
        println!("  diag truth#1 idx={} true_d={:.4} pq8_d={:.4}",
                 truth[0].0, truth[0].1, approx(truth[0].0));
        println!("  diag pq8#1   idx={} pq8_d={:.4}", pq8r[0].0, pq8r[0].1);
    }

    let recall_pq8: f32 = (0..q_count)
        .map(|q| recall_at_k(&pq8_top[q], &truth_all[q], k))
        .sum::<f32>() / q_count as f32;
    let recall_fs: f32 = (0..q_count)
        .map(|q| recall_at_k(&fs_top[q], &truth_all[q], k))
        .sum::<f32>() / q_count as f32;
    let recall_fsr: f32 = (0..q_count)
        .map(|q| recall_at_k(&fsr_top[q], &truth_all[q], k))
        .sum::<f32>() / q_count as f32;

    let qps = |elapsed: std::time::Duration| q_count as f64 / elapsed.as_secs_f64();
    println!("\n  variant            qps        total       recall@{}", k);
    println!("  flat (f32)      : {:>8.1}  {:>10?}  1.000", qps(t_flat), t_flat);
    println!("  pq8  (8-bit ADC): {:>8.1}  {:>10?}  {:.3}", qps(t_pq8), t_pq8, recall_pq8);
    println!("  fastscan-4 raw  : {:>8.1}  {:>10?}  {:.3}", qps(t_fs), t_fs, recall_fs);
    println!("  fastscan-4 + rr : {:>8.1}  {:>10?}  {:.3}  (rerank {} → {})",
             qps(t_fsr), t_fsr, recall_fsr, rerank_candidates, k);
    println!("\n  fastscan raw vs flat speedup: {:.1}x", t_flat.as_secs_f64() / t_fs.as_secs_f64());
    println!("  fastscan+rr vs flat speedup : {:.1}x", t_flat.as_secs_f64() / t_fsr.as_secs_f64());
    println!("  fastscan raw vs pq8  speedup: {:.1}x", t_pq8.as_secs_f64() / t_fs.as_secs_f64());
}
