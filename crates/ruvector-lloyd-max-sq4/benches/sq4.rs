//! Benchmark binary: no criterion dependency; prints throughput tables.
//!
//! Run with:
//!     cargo run --release -p ruvector-lloyd-max-sq4 --bench sq4

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use ruvector_lloyd_max_sq4::distance::{asym_l2_scan, build_query_lut};
use ruvector_lloyd_max_sq4::quant::{Codec, LloydMaxSQ4, UniformSQ4};
use std::time::Instant;

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

fn recall(
    codec: &dyn Codec,
    codes: &[u8],
    data: &[f32],
    queries: &[f32],
    n: usize,
    d: usize,
    nq: usize,
    k: usize,
) -> (f32, f32) {
    use std::collections::HashSet;
    let mut r = 0f32;
    let mut tot_dur = 0f64;
    let mut buf = Vec::new();
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let mut gt: Vec<(f32, u32)> = (0..n)
            .map(|j| (exact_l2(q, &data[j * d..(j + 1) * d]), j as u32))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt_set: HashSet<u32> = gt.iter().take(k).map(|p| p.1).collect();
        let t = Instant::now();
        asym_l2_scan(codec, q, codes, n, &mut buf);
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        tot_dur += t.elapsed().as_secs_f64();
        let hits = buf
            .iter()
            .take(k)
            .filter(|p| gt_set.contains(&p.1))
            .count();
        r += hits as f32 / k as f32;
    }
    (r / nq as f32, (tot_dur / nq as f64) as f32)
}

fn bench_dim(n: usize, d: usize, nq: usize, k: usize) {
    let data = gaussian(n, d, 1);
    let queries = gaussian(nq, d, 2);

    let t = Instant::now();
    let u = UniformSQ4::fit(&data, n, d);
    let fit_u = t.elapsed();
    let t = Instant::now();
    let l = LloydMaxSQ4::fit(&data, n, d, 25, 1e-5);
    let fit_l = t.elapsed();

    let t = Instant::now();
    let codes_u = u.encode_batch(&data, n);
    let enc_u = t.elapsed();
    let t = Instant::now();
    let codes_l = l.encode_batch(&data, n);
    let enc_l = t.elapsed();

    // Pure LUT scan throughput.
    let t = Instant::now();
    let lut = build_query_lut(&u, &queries[..d]);
    let mut buf = Vec::new();
    for _ in 0..10 {
        buf.clear();
        for i in 0..n {
            let code = &codes_u[i * u.code_bytes()..(i + 1) * u.code_bytes()];
            let dist = ruvector_lloyd_max_sq4::distance::asym_l2_lut(d, &lut, code);
            buf.push((dist, i as u32));
        }
    }
    let scan_u_total = t.elapsed();
    let scan_u_per_query = scan_u_total.as_secs_f64() / 10.0;

    let mse_u = mse(&u, &codes_u, &data, n, d);
    let mse_l = mse(&l, &codes_l, &data, n, d);
    let (r_u, lat_u) = recall(&u, &codes_u, &data, &queries, n, d, nq, k);
    let (r_l, lat_l) = recall(&l, &codes_l, &data, &queries, n, d, nq, k);

    let bytes_f32 = n * d * 4;
    let bytes_sq4 = n * u.code_bytes();
    let compression = bytes_f32 as f32 / bytes_sq4 as f32;

    println!(
        "n={:>6} d={:>3} | f32={:>5} KiB sq4={:>5} KiB (×{:.2}) | fit u={:>7.2?} l={:>7.2?} | enc u={:>7.2?} l={:>7.2?} | mse u={:.4} l={:.4} | r@{}: u={:.4} l={:.4} | scan {:.0} vec/µs | query lat u={:.1}µs l={:.1}µs",
        n, d,
        bytes_f32 / 1024, bytes_sq4 / 1024, compression,
        fit_u, fit_l, enc_u, enc_l,
        mse_u, mse_l,
        k, r_u, r_l,
        n as f64 / (scan_u_per_query * 1e6),
        lat_u * 1e6, lat_l * 1e6,
    );
}

fn main() {
    println!("# ruvector-lloyd-max-sq4 benchmark");
    println!("legend: scan = Uniform SQ4 LUT scan rate (vectors/µs). Larger is better.\n");
    bench_dim(5_000, 32, 100, 10);
    bench_dim(5_000, 64, 100, 10);
    bench_dim(5_000, 128, 100, 10);
    bench_dim(10_000, 64, 100, 10);
    bench_dim(20_000, 64, 100, 10);
}
