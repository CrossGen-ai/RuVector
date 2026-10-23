//! Simple bench binary (harness=false, no criterion dep): measures encode and
//! scan throughput for the three codecs on a skewed dataset. Emits real
//! cargo-run numbers.

use std::time::Instant;

use ruvector_hadamard_sq::distance::{brute_force_topk, codec_topk, make_skewed_dataset, recall};
use ruvector_hadamard_sq::quant::{Codec, HadamardSQ4, HadamardSQ8, UniformSQ4};

fn encode_bench<C: Codec>(c: &C, data: &[f32], dim: usize) -> (f64, usize) {
    let n = data.len() / dim;
    let stride = c.code_bytes();
    let mut codes = vec![0u8; n * stride];
    let t = Instant::now();
    for i in 0..n {
        c.encode(
            &data[i * dim..(i + 1) * dim],
            &mut codes[i * stride..(i + 1) * stride],
        );
    }
    let secs = t.elapsed().as_secs_f64();
    (n as f64 / secs, codes.len())
}

fn scan_bench<C: Codec>(c: &C, data: &[f32], queries: &[f32], dim: usize, k: usize) -> (f64, f32) {
    let n = data.len() / dim;
    let nq = queries.len() / dim;
    let stride = c.code_bytes();
    let mut codes = vec![0u8; n * stride];
    for i in 0..n {
        c.encode(
            &data[i * dim..(i + 1) * dim],
            &mut codes[i * stride..(i + 1) * stride],
        );
    }
    let t = Instant::now();
    let mut recall_total = 0.0f32;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let got = codec_topk(c, query, &codes, n, k);
        let truth = brute_force_topk(query, data, dim, k);
        recall_total += recall(&got, &truth);
    }
    let secs = t.elapsed().as_secs_f64();
    let qps = nq as f64 / secs;
    (qps, recall_total / nq as f32)
}

fn main() {
    let dim = 64;
    let n = 20_000;
    let nq = 100;
    let k = 10;
    let data = make_skewed_dataset(n, dim, 42);
    let queries = make_skewed_dataset(nq, dim, 1337);

    println!("ruvector-hadamard-sq bench");
    println!("  n={} nq={} d={} k={}", n, nq, dim, k);
    println!();

    let base = UniformSQ4::train(dim, &data);
    let h4 = HadamardSQ4::train(dim, &data, 7);
    let h8 = HadamardSQ8::train(dim, &data, 7);

    println!(
        "{:<16} {:>14} {:>14} {:>14} {:>12}",
        "codec", "encode(vec/s)", "scan(qps)", "scan(us/q)", "recall@10"
    );
    println!("{:-<74}", "");

    for c in &[("uniform-sq4", &base as &dyn Codec), ("hadamard-sq4", &h4 as &dyn Codec), ("hadamard-sq8", &h8 as &dyn Codec)] {
        let (enc_vps, _codes_bytes) = encode_bench_dyn(c.1, &data, dim);
        let (qps, r) = scan_bench_dyn(c.1, &data, &queries, dim, k);
        println!(
            "{:<16} {:>14.0} {:>14.1} {:>14.1} {:>12.3}",
            c.0,
            enc_vps,
            qps,
            1_000_000.0 / qps,
            r
        );
    }
    // Silence unused-fn warnings when all codecs go through the dyn path.
    let _ = encode_bench(&base, &data, dim);
    let _ = scan_bench(&base, &data, &queries, dim, k);
}

fn encode_bench_dyn(c: &dyn Codec, data: &[f32], dim: usize) -> (f64, usize) {
    let n = data.len() / dim;
    let stride = c.code_bytes();
    let mut codes = vec![0u8; n * stride];
    let t = Instant::now();
    for i in 0..n {
        c.encode(
            &data[i * dim..(i + 1) * dim],
            &mut codes[i * stride..(i + 1) * stride],
        );
    }
    let secs = t.elapsed().as_secs_f64();
    (n as f64 / secs, codes.len())
}

fn scan_bench_dyn(c: &dyn Codec, data: &[f32], queries: &[f32], dim: usize, k: usize) -> (f64, f32) {
    let n = data.len() / dim;
    let nq = queries.len() / dim;
    let stride = c.code_bytes();
    let mut codes = vec![0u8; n * stride];
    for i in 0..n {
        c.encode(
            &data[i * dim..(i + 1) * dim],
            &mut codes[i * stride..(i + 1) * stride],
        );
    }
    let t = Instant::now();
    let mut recall_total = 0.0f32;
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let got = codec_topk_dyn(c, query, &codes, n, k);
        let truth = brute_force_topk(query, data, dim, k);
        recall_total += recall(&got, &truth);
    }
    let secs = t.elapsed().as_secs_f64();
    let qps = nq as f64 / secs;
    (qps, recall_total / nq as f32)
}

fn codec_topk_dyn(c: &dyn Codec, query: &[f32], codes: &[u8], n: usize, k: usize) -> Vec<u32> {
    let stride = c.code_bytes();
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let d = c.l2_sq_asymmetric(query, &codes[i * stride..(i + 1) * stride]);
        heap.push((d, i as u32));
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap.truncate(k);
    heap.into_iter().map(|(_, id)| id).collect()
}
