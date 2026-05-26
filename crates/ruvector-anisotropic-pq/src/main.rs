//! `apq-bench`: end-to-end recall + memory benchmark for PQ vs Anisotropic PQ.
//!
//! Run: `cargo run --release -p ruvector-anisotropic-pq`
//!
//! Generates a synthetic dataset of unit-norm vectors with planted clusters,
//! trains three quantizers (PQ, APQ η=2, APQ η=4), encodes the database,
//! evaluates Recall@10 and Recall@100 vs brute-force MIPS ground truth,
//! and reports timings + bytes-per-vector.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_anisotropic_pq::{
    approx_topk, brute_force_topk, recall_at_k, ApqQuantizer, Pq, Quantizer,
};
use std::time::Instant;

fn make_clustered_unit_data(
    n: usize, d: usize, n_clusters: usize, seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng)).collect();
            normalize(&v)
        })
        .collect();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let c = &centers[rng.gen_range(0..n_clusters)];
        let noise: Vec<f32> = (0..d).map(|_| 0.15 * normal.sample(&mut rng)).collect();
        let v: Vec<f32> = c.iter().zip(noise).map(|(a, b)| a + b).collect();
        out.push(normalize(&v));
    }
    out
}

/// Realistic queries: pick a random db vector and perturb it slightly.
/// This models a "search for things similar to X" workload, which is where
/// score-aware quantization is supposed to win.
fn make_queries(db: &[Vec<f32>], n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    (0..n)
        .map(|_| {
            let base = &db[rng.gen_range(0..db.len())];
            let v: Vec<f32> = base
                .iter()
                .map(|a| a + 0.20 * normal.sample(&mut rng))
                .collect();
            normalize(&v)
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|a| a / n).collect()
}

fn evaluate<Q: Quantizer>(
    name: &str,
    q: &Q,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth_10: &[Vec<usize>],
    truth_100: &[Vec<usize>],
) {
    let t0 = Instant::now();
    let codes: Vec<Vec<u8>> = db.iter().map(|x| q.encode(x)).collect();
    let encode_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut r10 = 0.0_f32;
    let mut r100 = 0.0_f32;
    for (qi, query) in queries.iter().enumerate() {
        let a10 = approx_topk(q, query, &codes, 10);
        let a100 = approx_topk(q, query, &codes, 100);
        r10 += recall_at_k(&a10, &truth_10[qi]);
        r100 += recall_at_k(&a100, &truth_100[qi]);
    }
    let search_ms = t1.elapsed().as_secs_f64() * 1e3;
    r10 /= queries.len() as f32;
    r100 /= queries.len() as f32;

    let bpv = q.bytes_per_vector();
    let total_mb = (codes.len() * bpv) as f64 / (1024.0 * 1024.0);
    println!(
        "  {name:<14} | bpv={bpv:>3} | total={total_mb:>5.2} MiB | \
         encode={encode_ms:>7.1} ms | search={search_ms:>7.1} ms | \
         R@10={:.4} | R@100={:.4}",
        r10, r100,
    );
}

fn main() {
    let n = 20_000;
    let d = 128;
    let k = 256;       // K=256 → 1 byte per subcode
    let n_queries = 200;
    let n_clusters = 64;
    let train_iters = 12;
    let seed = 0xC0FFEE_u64;

    println!("ruvector-anisotropic-pq benchmark");
    println!(
        "  n={n}  d={d}  K={k}  queries={n_queries}  \
         clusters={n_clusters}  train_iters={train_iters}"
    );

    let t0 = Instant::now();
    let db = make_clustered_unit_data(n, d, n_clusters, seed);
    let queries = make_queries(&db, n_queries, d, seed ^ 0x1234);
    let _ = d; // silence unused if main reformats
    println!("  data gen: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    println!("  computing ground truth (brute-force MIPS)...");
    let t0 = Instant::now();
    let truth_10: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_topk(q, &db, 10))
        .collect();
    let truth_100: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_topk(q, &db, 100))
        .collect();
    println!("  ground truth: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    // Use first 8000 vectors for codebook training to keep training fast.
    let train: Vec<Vec<f32>> = db.iter().take(8_000).cloned().collect();
    let raw_mb = (n * d * 4) as f64 / (1024.0 * 1024.0);
    println!("  raw fp32 baseline storage: {raw_mb:.2} MiB");

    for &m in &[16_usize, 32] {
        println!();
        println!("--- M={m} subspaces (sub_dim={}) ---", d / m);

        let t0 = Instant::now();
        let pq = Pq::train(&train, m, k, train_iters, seed).unwrap();
        println!("  trained PQ            in {:>7.1} ms", t0.elapsed().as_secs_f64() * 1e3);

        let t0 = Instant::now();
        let apq2 = ApqQuantizer::train(&train, m, k, 2.0, train_iters, seed).unwrap();
        println!("  trained APQ (η=2.0)   in {:>7.1} ms", t0.elapsed().as_secs_f64() * 1e3);

        let t0 = Instant::now();
        let apq4 = ApqQuantizer::train(&train, m, k, 4.0, train_iters, seed).unwrap();
        println!("  trained APQ (η=4.0)   in {:>7.1} ms", t0.elapsed().as_secs_f64() * 1e3);

        println!();
        println!("  variant         | bpv | total mem | encode (full db) | search ({n_queries} q × 2 k) | recall");
        evaluate("PQ",        &pq,   &db, &queries, &truth_10, &truth_100);
        evaluate("APQ η=2.0", &apq2, &db, &queries, &truth_10, &truth_100);
        evaluate("APQ η=4.0", &apq4, &db, &queries, &truth_10, &truth_100);
        println!(
            "  compression ratio vs fp32: {:.1}×",
            (d * 4) as f32 / pq.bytes_per_vector() as f32
        );
    }
}
