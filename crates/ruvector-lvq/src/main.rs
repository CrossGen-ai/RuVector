//! End-to-end demo: build SQ8 / LVQ1-8 / LVQ1-4 / LVQ2-8x4 indexes over a
//! synthetic Gaussian dataset, measure recall@10, encode size, encode time,
//! and brute-force scan throughput. Prints a table consumed by the research
//! doc / gist.

use ruvector_lvq::{
    distance::Metric,
    quantizer::{total_bytes, Encoded, Quantizer},
    recall::{recall_at, topk_exact, topk_quantized},
    LvqOne, LvqTwo, Sq8,
};

use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;

const N: usize = 20_000;
const NQ: usize = 200;
const D: usize = 128;
const K: usize = 10;

fn gen_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut db = Vec::with_capacity(N);
    for _ in 0..N {
        let mut v = vec![0f32; D];
        for x in v.iter_mut() {
            let s: f32 = rng.sample(StandardNormal);
            *x = s;
        }
        db.push(v);
    }
    let mut q = Vec::with_capacity(NQ);
    for _ in 0..NQ {
        let mut v = vec![0f32; D];
        for x in v.iter_mut() {
            let s: f32 = rng.sample(StandardNormal);
            *x = s;
        }
        q.push(v);
    }
    (db, q)
}

fn truth(db: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    queries.iter().map(|q| topk_exact(q, db, K, Metric::L2)).collect()
}

fn run<Q: Quantizer>(name: &str, mut q: Q, db: &[Vec<f32>], queries: &[Vec<f32>], gt: &[Vec<usize>]) {
    let t = Instant::now();
    q.fit(db).unwrap();
    let fit_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let encoded: Vec<Encoded> = db.iter().map(|v| q.encode(v).unwrap()).collect();
    let enc_ms = t.elapsed().as_secs_f64() * 1000.0;

    // recall + scan throughput
    let mut recall_sum = 0f32;
    let t = Instant::now();
    for (i, qv) in queries.iter().enumerate() {
        let approx = topk_quantized(&q, qv, &encoded, K, Metric::L2);
        recall_sum += recall_at(&gt[i], &approx, K);
    }
    let scan_ms = t.elapsed().as_secs_f64() * 1000.0;
    let recall = recall_sum / queries.len() as f32;

    let per_vec_bytes = total_bytes(q.code_bytes());
    let total_mb = (per_vec_bytes * db.len()) as f64 / (1024.0 * 1024.0);
    let scans_per_sec = (queries.len() * db.len()) as f64 / (scan_ms / 1000.0);

    println!(
        "{:<10}  bits/comp={:>2}  bytes/vec={:>4}  index={:>6.2}MB  fit={:>6.1}ms  encode={:>7.1}ms  scan={:>7.1}ms  scans/s={:>10.0}  recall@{}={:.4}",
        name,
        q.bits_per_component(),
        per_vec_bytes,
        total_mb,
        fit_ms,
        enc_ms,
        scan_ms,
        scans_per_sec,
        K,
        recall
    );
}

fn main() {
    println!("ruvector-lvq demo  |  N={}  Nq={}  d={}  k={}", N, NQ, D, K);
    println!("Building dataset (Gaussian)...");
    let (db, queries) = gen_dataset();
    println!("Computing exact ground truth...");
    let gt = truth(&db, &queries);

    // fp32 baseline for reference
    {
        let t = Instant::now();
        let mut s = 0f32;
        for qv in &queries {
            for v in &db {
                s += ruvector_lvq::distance::l2_sq(qv, v);
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let per_vec_bytes = D * 4;
        let total_mb = (per_vec_bytes * db.len()) as f64 / (1024.0 * 1024.0);
        let scans_per_sec = (queries.len() * db.len()) as f64 / (ms / 1000.0);
        println!(
            "{:<10}  bits/comp=32  bytes/vec={:>4}  index={:>6.2}MB  fit=    n/a  encode=    n/a  scan={:>7.1}ms  scans/s={:>10.0}  recall@{}=1.0000  (sink={:.2})",
            "f32",
            per_vec_bytes,
            total_mb,
            ms,
            scans_per_sec,
            K,
            s
        );
    }

    run("SQ8",     Sq8::new(D),                          &db, &queries, &gt);
    run("LVQ1-8",  LvqOne::new(D, 8).unwrap(),           &db, &queries, &gt);
    run("LVQ1-4",  LvqOne::new(D, 4).unwrap(),           &db, &queries, &gt);
    run("LVQ2-8x4",LvqTwo::new(D, 8, 4).unwrap(),        &db, &queries, &gt);
}
