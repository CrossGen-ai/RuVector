//! End-to-end benchmark for `ruvector-rpq`.
//!
//! Trains three quantizers on a clustered synthetic dataset, encodes the
//! database, then measures encoding throughput, query throughput and
//! recall@10 against exact brute-force ground truth.
//!
//! Run:
//! ```text
//! cargo run --release -p ruvector-rpq --bin rpq-bench
//! ```

use ruvector_rpq::{
    brute_topk, recall_at_k, BenchRow, Pq, Quantizer, Rng, Rpq2, Rpq2Scorer, Sq8,
};
use std::time::Instant;

fn make_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    // Generate a mixture of Gaussians. Each cluster is picked uniformly and
    // has an isotropic std of 0.5 around a random unit-scale centroid.
    let mut rng = Rng::new(seed);
    // Cluster centroids.
    let mut centroids = vec![0.0f32; n_clusters * dim];
    for v in centroids.iter_mut() {
        *v = rng.next_gauss() * 3.0;
    }
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = (rng.next_u64() as usize) % n_clusters;
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dst = &mut out[i * dim..(i + 1) * dim];
        for d in 0..dim {
            dst[d] = cent[d] + rng.next_gauss() * 0.5;
        }
    }
    out
}

fn now_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn topk_from_scores(scores: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<(f32, u32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();
    idx.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    idx[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    idx.into_iter().take(k).map(|(_, i)| i).collect()
}

fn bench_pq_like<Q: Quantizer>(
    label: &'static str,
    q: &Q,
    db: &[f32],
    dim: usize,
    n_db: usize,
    queries: &[f32],
    n_q: usize,
    truth: &[Vec<u32>],
    k: usize,
    train_ms: f64,
) -> BenchRow {
    let cb = q.code_bytes();
    let mut codes = vec![0u8; n_db * cb];
    let t = Instant::now();
    for i in 0..n_db {
        q.encode(&db[i * dim..(i + 1) * dim], &mut codes[i * cb..(i + 1) * cb]);
    }
    let encode_ms = now_ms(t);

    let mut scores = vec![0.0f32; n_db];
    let mut recall_sum = 0.0f32;
    let t = Instant::now();
    for qi in 0..n_q {
        let qv = &queries[qi * dim..(qi + 1) * dim];
        // For Pq we can amortise LUT compute; for the trait fallback we
        // pay one LUT per item. This function accepts any Quantizer, and
        // the specialised paths are done elsewhere.
        for i in 0..n_db {
            scores[i] = q.adc_sq_distance(qv, &codes[i * cb..(i + 1) * cb]);
        }
        let pred = topk_from_scores(&scores, k);
        recall_sum += recall_at_k(&truth[qi], &pred);
    }
    let query_ms = now_ms(t) / n_q as f64;
    BenchRow {
        name: label,
        code_bytes: cb,
        train_ms,
        encode_ms,
        query_ms,
        recall_at_10: recall_sum / n_q as f32,
    }
}

/// Specialised path for `Pq` that reuses the query LUT across the database
/// sweep — the fair scoring cost for production use.
fn bench_pq_amortised(
    q: &Pq,
    db: &[f32],
    dim: usize,
    n_db: usize,
    queries: &[f32],
    n_q: usize,
    truth: &[Vec<u32>],
    k: usize,
    train_ms: f64,
) -> BenchRow {
    let cb = q.code_bytes();
    let mut codes = vec![0u8; n_db * cb];
    let t = Instant::now();
    for i in 0..n_db {
        q.encode_raw(&db[i * dim..(i + 1) * dim], &mut codes[i * cb..(i + 1) * cb]);
    }
    let encode_ms = now_ms(t);

    let mut scores = vec![0.0f32; n_db];
    let mut recall_sum = 0.0f32;
    let t = Instant::now();
    for qi in 0..n_q {
        let qv = &queries[qi * dim..(qi + 1) * dim];
        let lut = q.compute_lut(qv);
        for i in 0..n_db {
            scores[i] = q.adc_from_lut(&lut, &codes[i * cb..(i + 1) * cb]);
        }
        let pred = topk_from_scores(&scores, k);
        recall_sum += recall_at_k(&truth[qi], &pred);
    }
    let query_ms = now_ms(t) / n_q as f64;
    BenchRow {
        name: "pq (amortised)",
        code_bytes: cb,
        train_ms,
        encode_ms,
        query_ms,
        recall_at_10: recall_sum / n_q as f32,
    }
}

/// Amortised RPQ2 scorer path (bucket-by-coarse-code).
fn bench_rpq_amortised(
    rpq: &Rpq2,
    db: &[f32],
    dim: usize,
    n_db: usize,
    queries: &[f32],
    n_q: usize,
    truth: &[Vec<u32>],
    k: usize,
    train_ms: f64,
) -> BenchRow {
    let cb = rpq.code_bytes();
    let mut codes = vec![0u8; n_db * cb];
    let t = Instant::now();
    for i in 0..n_db {
        rpq.encode(&db[i * dim..(i + 1) * dim], &mut codes[i * cb..(i + 1) * cb]);
    }
    let encode_ms = now_ms(t);

    let scorer = Rpq2Scorer::new(rpq, &codes);
    let mut scores = vec![0.0f32; n_db];
    let mut recall_sum = 0.0f32;
    let t = Instant::now();
    for qi in 0..n_q {
        let qv = &queries[qi * dim..(qi + 1) * dim];
        scorer.score_all(qv, &mut scores);
        let pred = topk_from_scores(&scores, k);
        recall_sum += recall_at_k(&truth[qi], &pred);
    }
    let query_ms = now_ms(t) / n_q as f64;
    BenchRow {
        name: "rpq2 (amort)",
        code_bytes: cb,
        train_ms,
        encode_ms,
        query_ms,
        recall_at_10: recall_sum / n_q as f32,
    }
}

fn main() {
    // Modest sizes so `cargo run --release` completes in seconds while still
    // producing statistically meaningful recall numbers.
    let dim: usize = 64;
    let n_train: usize = 10_000;
    let n_db: usize = 20_000;
    let n_q: usize = 200;
    let k: usize = 10;
    // Fair comparison: PQ-16 and RPQ2 both use 16 bytes/vector.
    // PQ-8 is included as a half-storage compression baseline.
    let m_pq_small: usize = 8; //  8 bytes / vector, subdim=8
    let m_pq_fair: usize = 16; // 16 bytes / vector, subdim=4
    let m1_rpq: usize = 8; //   coarse: 8 bytes, subdim=8
    let m2_rpq: usize = 8; // residual: 8 bytes, subdim=8 (16 B total)
    let iters: usize = 25;

    println!(
        "ruvector-rpq bench: dim={dim} n_train={n_train} n_db={n_db} n_q={n_q} k={k} pq_small={m_pq_small} pq_fair={m_pq_fair} rpq2=({m1_rpq},{m2_rpq}) iters={iters}"
    );

    let mut rng = Rng::new(0xC0FFEE);
    let train = make_clustered(n_train, dim, 64, 0xAAAA);
    let db = make_clustered(n_db, dim, 64, 0xBBBB);
    let queries = make_clustered(n_q, dim, 64, 0xCCCC);

    // Ground truth (brute-force).
    println!("[gt] computing brute-force top-{k} for {n_q} queries over {n_db} database vectors...");
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = (0..n_q)
        .map(|qi| brute_topk(&queries[qi * dim..(qi + 1) * dim], &db, dim, k))
        .collect();
    let gt_ms = now_ms(t);
    println!("[gt] done in {:.1} ms ({:.2} ms/q)", gt_ms, gt_ms / n_q as f64);

    // Train the three quantizers.
    println!("[train] PQ  (m={m_pq_small})...");
    let t = Instant::now();
    let pq_small = Pq::train(&train, dim, m_pq_small, iters, &mut rng);
    let pq_small_train_ms = now_ms(t);

    println!("[train] PQ  (m={m_pq_fair})...");
    let t = Instant::now();
    let pq_fair = Pq::train(&train, dim, m_pq_fair, iters, &mut rng);
    let pq_fair_train_ms = now_ms(t);

    println!("[train] RPQ2 (m1={m1_rpq}, m2={m2_rpq})...");
    let t = Instant::now();
    let rpq = Rpq2::train(&train, dim, m1_rpq, m2_rpq, iters, &mut rng);
    let rpq_train_ms = now_ms(t);

    println!("[train] SQ8...");
    let t = Instant::now();
    let sq = Sq8::train(&train, dim);
    let sq_train_ms = now_ms(t);

    // Run bench rows.
    let mut rows = Vec::new();
    let mut r = bench_pq_amortised(&pq_small, &db, dim, n_db, &queries, n_q, &truth, k, pq_small_train_ms);
    r.name = "pq-8";
    rows.push(r);
    let mut r = bench_pq_amortised(&pq_fair, &db, dim, n_db, &queries, n_q, &truth, k, pq_fair_train_ms);
    r.name = "pq-16";
    rows.push(r);
    rows.push(bench_rpq_amortised(&rpq, &db, dim, n_db, &queries, n_q, &truth, k, rpq_train_ms));
    rows.push(bench_pq_like("sq8", &sq, &db, dim, n_db, &queries, n_q, &truth, k, sq_train_ms));

    println!();
    println!("Results (dim={dim}, n_db={n_db}, n_q={n_q}, k={k}):");
    println!("---------------------------------------------------------------------------------------------------");
    for r in &rows {
        println!("{r}");
    }
    println!("---------------------------------------------------------------------------------------------------");
    // Memory footprint summary.
    let raw_mb = (n_db * dim * 4) as f64 / (1024.0 * 1024.0);
    println!("raw f32 database: {:.2} MiB", raw_mb);
    for r in &rows {
        let mb = (n_db * r.code_bytes) as f64 / (1024.0 * 1024.0);
        let ratio = (dim * 4) as f64 / r.code_bytes as f64;
        println!(
            "  {:<15} -> {:>7.2} MiB (compression {:.1}x)",
            r.name, mb, ratio
        );
    }
}
