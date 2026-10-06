//! Demo: train both coders on a synthetic dataset, print MSE and recall@10.
//!
//! Run with:
//!     cargo run --release -p ruvector-lloyd-max-sq4 --example demo

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use ruvector_lloyd_max_sq4::distance::{asym_l2_scan, build_query_lut, recall_at_k};
use ruvector_lloyd_max_sq4::quant::{Codec, LloydMaxSQ4, UniformSQ4};

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n * d)
        .map(|_| {
            let x: f32 = StandardNormal.sample(&mut rng);
            x
        })
        .collect()
}

fn exact_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn mse(codec: &dyn Codec, codes: &[u8], data: &[f32], n: usize, d: usize) -> f32 {
    let cb = codec.code_bytes();
    let mut buf = vec![0f32; d];
    let mut acc = 0f32;
    for i in 0..n {
        codec.decode_into(&codes[i * cb..(i + 1) * cb], &mut buf);
        for j in 0..d {
            let e = buf[j] - data[i * d + j];
            acc += e * e;
        }
    }
    acc / (n * d) as f32
}

fn recall_run(codec: &dyn Codec, codes: &[u8], data: &[f32], queries: &[f32], n: usize, d: usize, nq: usize, k: usize) -> f32 {
    let mut r = 0f32;
    let mut buf = Vec::new();
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let mut gt: Vec<(f32, u32)> = (0..n)
            .map(|j| (exact_l2(q, &data[j * d..(j + 1) * d]), j as u32))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt_ids: Vec<u32> = gt.iter().take(k).map(|p| p.1).collect();

        // Use the LUT scan path.
        let _ = build_query_lut(codec, q);
        asym_l2_scan(codec, q, codes, n, &mut buf);
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let cand: Vec<u32> = buf.iter().take(k).map(|p| p.1).collect();
        r += recall_at_k(k, &gt_ids, &cand);
    }
    r / nq as f32
}

fn main() {
    let n = 10_000;
    let d = 64;
    let nq = 200;
    let k = 10;

    println!("# ruvector-lloyd-max-sq4 demo");
    println!("Dataset: Gaussian(0, I), n={}, d={}, queries={}, k={}", n, d, nq, k);

    let data = gaussian(n, d, 1);
    let queries = gaussian(nq, d, 2);

    let t_fit_u = std::time::Instant::now();
    let codec_u = UniformSQ4::fit(&data, n, d);
    let t_fit_u = t_fit_u.elapsed();

    let t_fit_l = std::time::Instant::now();
    let codec_l = LloydMaxSQ4::fit(&data, n, d, 25, 1e-5);
    let t_fit_l = t_fit_l.elapsed();

    let codes_u = codec_u.encode_batch(&data, n);
    let codes_l = codec_l.encode_batch(&data, n);

    let bytes = codec_u.code_bytes() * n;
    let bytes_f32 = n * d * 4;
    println!(
        "Memory: f32={} MiB, sq4={} MiB (compression {:.1}x)",
        bytes_f32 / (1024 * 1024),
        bytes / (1024 * 1024).max(1),
        bytes_f32 as f32 / bytes as f32
    );
    println!("Fit time: uniform={:?}, lloyd={:?}", t_fit_u, t_fit_l);

    let mse_u = mse(&codec_u, &codes_u, &data, n, d);
    let mse_l = mse(&codec_l, &codes_l, &data, n, d);
    println!("Reconstruction MSE: uniform={:.5}, lloyd={:.5}", mse_u, mse_l);

    let t = std::time::Instant::now();
    let r_u = recall_run(&codec_u, &codes_u, &data, &queries, n, d, nq, k);
    let dt_u = t.elapsed();
    let t = std::time::Instant::now();
    let r_l = recall_run(&codec_l, &codes_l, &data, &queries, n, d, nq, k);
    let dt_l = t.elapsed();

    println!("Recall@{}: uniform={:.4} ({:?}), lloyd={:.4} ({:?})",
        k, r_u, dt_u, r_l, dt_l);
}
