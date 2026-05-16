//! Benchmark runner producing the numbers used in
//! `docs/research/nightly/2026-05-16-anisotropic-vq/README.md`.
//!
//! Runs three quantizers (PQ, APQ, OPQ+APQ) over a synthetic 128-d unit-norm
//! mixture-of-Gaussians dataset and prints recall@10 + per-query latency.

use ruvector_anisotropic_pq::{
    distance::{topk_by_dot, topk_exact_dot},
    recall_at_k, synthetic, Apq, OpqApq, Pq, Quantizer,
};
use std::time::Instant;

fn main() {
    let dim = 128;
    let n_train = 20_000;
    let n_query = 500;
    let n_clusters = 64;
    let max_iter = 25;
    let top_k = 10;
    let seed = 0x5141_2026_u64;
    // Default config (m=16, eta=4) — additional sweeps below.
    let m = 16;          // 16 subspaces, sub_dim = 8
    let k = 256;         // 1 byte / code
    let eta = 4.0;       // anisotropic weight ratio

    println!("anisotropic-vq nightly bench");
    println!(
        "  dim={}  n_train={}  n_query={}  m={}  k={}  eta={}  iter={}  top_k={}",
        dim, n_train, n_query, m, k, eta, max_iter, top_k
    );
    println!();

    let t0 = Instant::now();
    let ds = synthetic::make(n_train, n_query, dim, n_clusters, seed);
    println!("dataset generated in {:.2?}", t0.elapsed());

    // Ground truth.
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = ds
        .queries
        .iter()
        .map(|q| topk_exact_dot(&ds.train, q, top_k))
        .collect();
    let gt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "ground truth (brute force IP): {:.2?}  ({:.3} ms/query)",
        t0.elapsed(),
        gt_ms / n_query as f64
    );
    println!();

    println!("{:<14} {:>12} {:>16} {:>14} {:>14}", "variant", "train (ms)", "encode all (ms)", "us/query", "recall@10");
    println!("{}", "-".repeat(74));

    // ---- PQ baseline ----
    let t0 = Instant::now();
    let pq = Pq::train(&ds.train, m, k, max_iter, seed).expect("pq train");
    let pq_train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let pq_codes: Vec<Vec<u8>> = pq.encode_batch(&ds.train);
    let pq_enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (pq_recall, pq_us) = score(&pq, &pq_codes, &ds.queries, &truth, top_k);
    println!(
        "{:<14} {:>12.1} {:>16.1} {:>14.2} {:>14.4}",
        "PQ (eta=1)", pq_train_ms, pq_enc_ms, pq_us, pq_recall
    );

    // ---- Anisotropic PQ ----
    let t0 = Instant::now();
    let apq = Apq::train(&ds.train, m, k, eta, max_iter, seed).expect("apq train");
    let apq_train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let apq_codes: Vec<Vec<u8>> = apq.encode_batch(&ds.train);
    let apq_enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (apq_recall, apq_us) = score(&apq, &apq_codes, &ds.queries, &truth, top_k);
    println!(
        "{:<14} {:>12.1} {:>16.1} {:>14.2} {:>14.4}",
        format!("APQ (eta={})", eta), apq_train_ms, apq_enc_ms, apq_us, apq_recall
    );

    // ---- OPQ + APQ ----
    let t0 = Instant::now();
    let opq = OpqApq::train(&ds.train, m, k, eta, max_iter, seed).expect("opq train");
    let opq_train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    let opq_codes: Vec<Vec<u8>> = opq.encode_batch(&ds.train);
    let opq_enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (opq_recall, opq_us) = score(&opq, &opq_codes, &ds.queries, &truth, top_k);
    println!(
        "{:<14} {:>12.1} {:>16.1} {:>14.2} {:>14.4}",
        "OPQ+APQ", opq_train_ms, opq_enc_ms, opq_us, opq_recall
    );

    println!();
    println!("Memory per vector: {} bytes (vs {} bytes raw fp32) -> {:.0}x smaller",
        m, dim * 4, (dim * 4) as f32 / m as f32);

    println!();
    println!("Delta recall@10:");
    println!("  APQ vs PQ:     {:+.4}", apq_recall - pq_recall);
    println!("  OPQ+APQ vs PQ: {:+.4}", opq_recall - pq_recall);

    println!();
    println!("=== Sweep: eta (m=16, k=256, dim=128) ===");
    println!("{:<8} {:>14}", "eta", "recall@10");
    println!("{}", "-".repeat(24));
    let mut best_apq_recall = pq_recall;
    let mut best_eta = 1.0f32;
    for &e in &[1.0f32, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0] {
        let q = Apq::train(&ds.train, m, k, e, max_iter, seed).expect("apq");
        let codes = q.encode_batch(&ds.train);
        let (r, _us) = score(&q, &codes, &ds.queries, &truth, top_k);
        println!("{:<8.2} {:>14.4}", e, r);
        if r > best_apq_recall { best_apq_recall = r; best_eta = e; }
    }
    println!("best eta = {} (recall@10 = {:.4})", best_eta, best_apq_recall);

    println!();
    println!("=== Sweep: m (k=256, dim=128, eta={}) ===", best_eta);
    println!("{:<6} {:>14} {:>14} {:>14}", "m", "bytes/vec", "PQ r@10", "APQ r@10");
    println!("{}", "-".repeat(50));
    for &mm in &[8usize, 16, 32] {
        if dim % mm != 0 { continue; }
        let pq2 = Pq::train(&ds.train, mm, k, max_iter, seed).unwrap();
        let apq2 = Apq::train(&ds.train, mm, k, best_eta, max_iter, seed).unwrap();
        let cp = pq2.encode_batch(&ds.train);
        let ca = apq2.encode_batch(&ds.train);
        let (rp, _) = score(&pq2, &cp, &ds.queries, &truth, top_k);
        let (ra, _) = score(&apq2, &ca, &ds.queries, &truth, top_k);
        println!("{:<6} {:>14} {:>14.4} {:>14.4}", mm, mm, rp, ra);
    }
}

fn score<Q: Quantizer>(
    q: &Q,
    codes: &[Vec<u8>],
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    top_k: usize,
) -> (f32, f64) {
    let t0 = Instant::now();
    let preds: Vec<Vec<u32>> = queries
        .iter()
        .map(|qv| topk_by_dot(q, qv, codes, top_k))
        .collect();
    let us = t0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    let r = recall_at_k(&preds, truth, top_k);
    (r, us)
}
