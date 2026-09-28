//! abpq-demo — runnable benchmark comparing AB-PQ vs uniform PQ vs FP32 truth.
//!
//! Prints reconstruction MSE, recall@10, and encode+search wall time on a
//! synthetic anisotropic dataset. All numbers are real cargo-run outputs.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_abpq::{pack, AbPq, AbPqConfig};
use std::time::Instant;

fn gen(n: usize, d: usize, seed: u64, decay: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scales: Vec<f32> = (0..d).map(|j| 1.0 / (1.0 + j as f32 / decay)).collect();
    let mut out = vec![0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let normal = Normal::new(0.0, scales[j]).unwrap();
            out[i * d + j] = normal.sample(&mut rng) as f32;
        }
    }
    out
}

fn recall_at_10(pq: &AbPq, data: &[f32], codes: &[u8], queries: &[f32]) -> f32 {
    let d = pq.cfg.d;
    let n = data.len() / d;
    let nq = queries.len() / d;
    let cb = pq.code_bytes();
    let mut hits = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let mut gt: Vec<(f32, usize)> = (0..n).map(|i| {
            let v = &data[i * d..(i + 1) * d];
            let mut s = 0f32;
            for j in 0..d { let df = q[j] - v[j]; s += df * df; }
            (s, i)
        }).collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt_set: std::collections::HashSet<usize> =
            gt.iter().take(10).map(|(_, i)| *i).collect();
        let tbl = pq.build_adc_table(q);
        let mut est: Vec<(f32, usize)> = (0..n).map(|i| {
            (pq.adc_distance(&codes[i * cb..(i + 1) * cb], &tbl), i)
        }).collect();
        est.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        hits += est.iter().take(10).filter(|(_, i)| gt_set.contains(i)).count();
    }
    hits as f32 / (nq * 10) as f32
}

fn mse(pq: &AbPq, data: &[f32]) -> f32 {
    let d = pq.cfg.d;
    let m = pq.cfg.m;
    let dsub = d / m;
    let n = data.len() / d;
    let mut buf = vec![0u8; pq.code_bytes()];
    let mut e = 0f32;
    for i in 0..n {
        let v = &data[i * d..(i + 1) * d];
        pq.encode_into(v, &mut buf);
        for s in 0..m {
            let code = pack::read_bits(&buf, pq.bit_offsets[s], pq.bits[s]) as u16;
            let cent = pq.subs[s].centroid(code);
            for j in 0..dsub {
                let diff = v[s * dsub + j] - cent[j];
                e += diff * diff;
            }
        }
    }
    e / n as f32
}

fn bench_one(label: &str, pq: &AbPq, data: &[f32], queries: &[f32]) {
    let t0 = Instant::now();
    let codes = pq.encode_batch(data);
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let r = recall_at_10(pq, data, &codes, queries);
    let search_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let m = mse(pq, data);
    let bits: Vec<String> = pq.bits.iter().map(|b| b.to_string()).collect();
    println!(
        "{:<24} bits=[{}]  code_bytes={:>2}  MSE={:.4}  recall@10={:.3}  encode={:>6.1}ms  search={:>6.1}ms",
        label, bits.join(","), pq.code_bytes(), m, r, enc_ms, search_ms
    );
}

fn main() {
    let (n, d, m, nq) = (5000, 64, 16, 200);
    println!("AB-PQ demo — n={} d={} m={} queries={}", n, d, m, nq);
    let data = gen(n, d, 11, 6.0);
    let queries = gen(nq, d, 22, 6.0);

    // AB-PQ (avg 4 bits)
    let mut cfg_ab = AbPqConfig::new(d, m, 4);
    cfg_ab.seed = 100;
    let ab = AbPq::train(cfg_ab, &data).expect("train ab");
    // Uniform baseline (4 bits everywhere)
    let mut cfg_u = AbPqConfig::new(d, m, 4);
    cfg_u.min_bits = 4; cfg_u.max_bits = 4; cfg_u.seed = 100;
    let u = AbPq::train(cfg_u, &data).expect("train uni");
    // Coarser uniform: 3 bits (fewer bytes; distortion baseline for smaller budget)
    let mut cfg_u3 = AbPqConfig::new(d, m, 3);
    cfg_u3.min_bits = 3; cfg_u3.max_bits = 3; cfg_u3.seed = 100;
    let u3 = AbPq::train(cfg_u3, &data).expect("train uni3");

    println!("{}", "-".repeat(112));
    bench_one("uniform PQ (3 bits)", &u3, &data, &queries);
    bench_one("uniform PQ (4 bits)", &u, &data, &queries);
    bench_one("AB-PQ (avg 4 bits)", &ab, &data, &queries);
    println!("{}", "-".repeat(112));
}
