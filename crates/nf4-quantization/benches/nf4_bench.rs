//! Real-numbers benchmark driver for NF4 variants vs linear SQ-4 and FP32.
//!
//! Metrics:
//!   - reconstruction MSE
//:   - recall@10 on exact-NN ground truth from the FP32 dataset
//:   - encode + scan throughput (vectors/second and ns/distance)
//!
//! Dataset: synthetic GloVe-like mixture — Gaussian core with heavy
//! tails and per-cluster bias, enough to distinguish fixed / asymmetric
//! / learned codebooks without requiring an external download.

use nf4_quantization::quantizer::{Code4, Quantizer};
use nf4_quantization::{AsymmetricNf4, FixedNf4, LearnedNf4};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::time::Instant;

const N_BASE: usize = 10_000;
const N_QUERY: usize = 200;
const DIM: usize = 128;
const K: usize = 10;

/// Linear SQ-4 (uniform steps on a per-vector max-abs scale) — the
/// classical 4-bit scalar quantizer we need to beat.
struct Sq4;

impl Quantizer for Sq4 {
    fn encode(&self, x: &[f32]) -> Code4 {
        let m = x.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-12);
        let mut c = Code4::new(x.len(), m, 0.0);
        let inv = 1.0 / m;
        for (i, &xi) in x.iter().enumerate() {
            // Map to [-1, 1] then to {-7.5, -6.5, ..., 7.5} / 7.5
            let n = (xi * inv).clamp(-1.0, 1.0);
            let q = ((n * 7.5 + 7.5).round() as i32).clamp(0, 15) as u8;
            c.set_nibble(i, q);
        }
        c
    }
    fn decode_into(&self, code: &Code4, out: &mut [f32]) {
        for i in 0..code.dim as usize {
            let k = code.get_nibble(i) as f32;
            out[i] = code.scale * ((k - 7.5) / 7.5);
        }
    }
    #[inline]
    fn level_at(&self, k: u8) -> f32 {
        ((k & 0x0F) as f32 - 7.5) / 7.5
    }
}

fn gen_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let cluster_count = 8;
    let centers: Vec<Vec<f32>> = (0..cluster_count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let s: f32 = StandardNormal.sample(&mut rng);
                    s * 0.5
                })
                .collect()
        })
        .collect();
    let gauss = Normal::new(0f32, 0.35).unwrap();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..cluster_count)];
            let heavy = rng.gen_bool(0.05); // 5% heavy-tailed
            (0..dim)
                .map(|j| {
                    let noise = if heavy {
                        gauss.sample(&mut rng) * 3.0
                    } else {
                        gauss.sample(&mut rng)
                    };
                    c[j] + noise
                })
                .collect()
        })
        .collect()
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn topk<F>(n: usize, k: usize, mut f: F) -> Vec<usize>
where
    F: FnMut(usize) -> f32,
{
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for i in 0..n {
        let d = f(i);
        if heap.len() < k {
            heap.push((d, i));
            heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } else if d < heap[0].0 {
            heap[0] = (d, i);
            heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap.into_iter().map(|(_, i)| i).collect()
}

fn recall_at_k(gt: &[Vec<usize>], got: &[Vec<usize>], k: usize) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (g, a) in gt.iter().zip(got.iter()) {
        for &gi in g.iter().take(k) {
            if a.iter().take(k).any(|&ai| ai == gi) {
                hits += 1;
            }
            total += 1;
        }
    }
    hits as f32 / total.max(1) as f32
}

fn eval<Q: Quantizer>(
    name: &str,
    q: &Q,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
) {
    // Encode all base vectors; measure encode throughput.
    let t0 = Instant::now();
    let codes: Vec<Code4> = base.iter().map(|v| q.encode(v)).collect();
    let enc_elapsed = t0.elapsed();

    // Reconstruction MSE.
    let mut buf = vec![0f32; base[0].len()];
    let mut mse_acc = 0f64;
    let mut n_scalars = 0u64;
    for (v, c) in base.iter().zip(codes.iter()) {
        q.decode_into(c, &mut buf);
        for i in 0..v.len() {
            mse_acc += (v[i] as f64 - buf[i] as f64).powi(2);
            n_scalars += 1;
        }
    }
    let mse = mse_acc / n_scalars as f64;

    // Search pass + measure scan time.
    let t1 = Instant::now();
    let mut results = Vec::with_capacity(queries.len());
    for qv in queries {
        let r = topk(codes.len(), K, |i| q.asym_l2_sq(qv, &codes[i]));
        results.push(r);
    }
    let scan_elapsed = t1.elapsed();
    let dists = (queries.len() * codes.len()) as f64;
    let ns_per_dist = scan_elapsed.as_nanos() as f64 / dists;
    let r10 = recall_at_k(gt, &results, K);

    let code_bytes = codes[0].bytes.len() + 8; // bytes + scale(+bias when relevant)
    let fp32_bytes = base[0].len() * 4;
    let ratio = fp32_bytes as f32 / code_bytes as f32;
    println!(
        "{:<20} | recall@10 = {:.4} | MSE = {:.6} | encode {:>7.2} Mvec/s | ns/dist {:>6.1} | {:>5.1}x smaller ({} vs {} B)",
        name,
        r10,
        mse,
        base.len() as f64 / enc_elapsed.as_secs_f64() / 1e6,
        ns_per_dist,
        ratio,
        code_bytes,
        fp32_bytes,
    );
}

fn main() {
    println!("== NF4 Quantization Benchmark ==");
    println!(
        "dataset: n={} base, {} queries, dim={}, synthetic 8-cluster Gaussian + 5% heavy tails",
        N_BASE, N_QUERY, DIM
    );

    let base = gen_dataset(N_BASE, DIM, 42);
    let queries = gen_dataset(N_QUERY, DIM, 1337);

    // FP32 ground truth top-K.
    let t0 = Instant::now();
    let gt: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| topk(base.len(), K, |i| l2_sq(q, &base[i])))
        .collect();
    let fp32_scan = t0.elapsed();
    let ns_fp32 =
        fp32_scan.as_nanos() as f64 / (queries.len() * base.len()) as f64;
    println!(
        "\nfp32 scan (reference): ns/dist = {:.1}, raw size = {} B / vec",
        ns_fp32,
        DIM * 4
    );
    println!();

    // Build quantizers.
    let flat: Vec<f32> = base.iter().flatten().copied().collect();
    let fixed = FixedNf4::fit_global_scale(&flat);
    let asym = AsymmetricNf4::new();

    // For learned, fit on a per-vector min/max-normalized sample so the
    // codebook sees the same distribution the encoder will see.
    let mut rng = StdRng::seed_from_u64(99);
    let sample_size = 50_000;
    let mut sample: Vec<f32> = Vec::with_capacity(sample_size);
    while sample.len() < sample_size {
        let v = &base[rng.gen_range(0..base.len())];
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let span = (mx - mn).max(1e-12);
        let scale = span * 0.5;
        let bias = (mn + mx) * 0.5;
        for &x in v {
            sample.push(((x - bias) / scale).clamp(-1.0, 1.0));
        }
    }
    let learned = LearnedNf4::fit(&sample, 25);

    println!(
        "{:<20} | {:>12} | {:>8} | {:>14} | {:>8} | compression",
        "variant", "recall@10", "MSE", "encode Mvec/s", "ns/dist"
    );
    println!("{}", "-".repeat(110));
    eval("SQ-4 (uniform)", &Sq4, &base, &queries, &gt);
    eval("NF4-fixed", &fixed, &base, &queries, &gt);
    eval("NF4-asymmetric", &asym, &base, &queries, &gt);
    eval("NF4-learned", &learned, &base, &queries, &gt);
}
