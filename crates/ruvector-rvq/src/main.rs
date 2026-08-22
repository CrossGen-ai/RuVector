//! `rvq-demo`: end-to-end sanity run that trains an RVQ, indexes a
//! synthetic corpus, and reports compression + recall@10 + timings on
//! real (not aspirational) numbers.
//!
//! Run: `cargo run --release -p ruvector-rvq --bin rvq-demo`

use ruvector_rvq::{Rvq, RvqConfig, RvqIndex};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn brute_l2_topk(data: &[f32], d: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / d;
    let mut scored: Vec<(u32, f32)> = (0..n).map(|i| {
        let mut s = 0f32;
        for j in 0..d { let e = data[i * d + j] - q[j]; s += e * e; }
        (i as u32, s)
    }).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn main() {
    let d = 128;
    let n = 10_000;
    let n_queries = 200;
    let k = 10;

    println!("=== RVQ demo: n={n}, d={d}, queries={n_queries}, k={k} ===\n");

    let data = synth(n, d, 1);
    let queries = synth(n_queries, d, 2);

    for &stages in &[4usize, 8, 16] {
        let cfg = RvqConfig { stages, k: 256, kmeans_iters: 12, seed: 42 };
        let t0 = Instant::now();
        let rvq = Rvq::train(&data, n, d, &cfg).unwrap();
        let train_ms = t0.elapsed().as_millis();

        let t1 = Instant::now();
        let idx = RvqIndex::build(rvq, &data);
        let build_ms = t1.elapsed().as_millis();

        // recall
        let mut hits = 0usize;
        let t2 = Instant::now();
        for qi in 0..n_queries {
            let q = &queries[qi * d..(qi + 1) * d];
            let truth: std::collections::HashSet<u32> =
                brute_l2_topk(&data, d, q, k).into_iter().collect();
            let got = idx.search_l2(q, k);
            for r in &got { if truth.contains(&r.id) { hits += 1; } }
        }
        let qmicros = t2.elapsed().as_micros() as f64 / n_queries as f64;
        let recall = hits as f32 / (n_queries * k) as f32;
        let raw_bytes = n * d * 4;
        let comp_bytes = idx.code_bytes();
        println!(
            "stages={stages:>2}  train={train_ms:>5}ms  build={build_ms:>4}ms  \
             raw={raw_bytes} B  compressed={comp_bytes} B  ratio={:.1}x  \
             recall@{k}={recall:.3}  query≈{qmicros:.0} µs (incl. ground-truth scan)",
             raw_bytes as f32 / comp_bytes as f32
        );
    }

    // Direct scan of RVQ only (no ground-truth per query) to measure the
    // actual search hot path.
    println!("\n--- Pure-scan timings (no ground-truth in loop) ---");
    for &stages in &[4usize, 8, 16] {
        let cfg = RvqConfig { stages, k: 256, kmeans_iters: 12, seed: 42 };
        let rvq = Rvq::train(&data, n, d, &cfg).unwrap();
        let idx = RvqIndex::build(rvq, &data);
        let t = Instant::now();
        let mut checksum = 0f32;
        for qi in 0..n_queries {
            let q = &queries[qi * d..(qi + 1) * d];
            let top = idx.search_l2(q, k);
            checksum += top[0].score;
        }
        let per = t.elapsed().as_micros() as f64 / n_queries as f64;
        println!("stages={stages:>2}  {per:>6.1} µs/query  (checksum={checksum:.2})");
    }

    // Baseline: brute-force f32 L2.
    let t = Instant::now();
    let mut checksum = 0f32;
    for qi in 0..n_queries {
        let q = &queries[qi * d..(qi + 1) * d];
        let top = brute_l2_topk(&data, d, q, k);
        checksum += top[0] as f32;
    }
    let per = t.elapsed().as_micros() as f64 / n_queries as f64;
    println!("baseline flat-f32    {per:>6.1} µs/query  (checksum={checksum:.2})");
}
