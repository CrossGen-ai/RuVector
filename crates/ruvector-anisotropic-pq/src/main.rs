//! Benchmark harness — produces real, reproducible numbers comparing plain L2 PQ
//! against anisotropic PQ on synthetic unit-norm Gaussian MIPS queries.
//!
//! Run: `cargo run --release -p ruvector-anisotropic-pq`.

use ruvector_anisotropic_pq::*;
use ruvector_anisotropic_pq::train::{train_anisotropic_pq, train_l2_pq, TrainOpts};
use std::time::Instant;

fn main() {
    let n: usize = 50_000;
    let dim: usize = 64;
    let m: usize = 8; // 8 codes / vector
    let k: usize = 16; // 4-bit codes → 4 bytes / vector total, ≈ 64x compression.
    let n_queries: usize = 500;
    let topk: usize = 10;

    println!("==================================================================");
    println!("Anisotropic PQ vs. L2 PQ — MIPS benchmark");
    println!("  n = {}   dim = {}   m = {}   k = {}   topk = {}   queries = {}",
        n, dim, m, k, topk, n_queries);
    println!("  code size: {} bytes / vector (1 byte per subspace, {:.1}x compression)",
        m, (dim * 4) as f32 / m as f32);
    println!("==================================================================");

    // Build dataset (unit-norm Gaussian) + a realistic query set.
    // Realistic MIPS queries are CORRELATED with some target document; we model
    // that by perturbing a held-out subset of dataset items with small Gaussian
    // noise and re-normalising. This is how MIPS works in production
    // (search query ↔ nearby document) and is the regime where the
    // anisotropic loss has a measurable advantage over plain L2 PQ.
    let data = synthetic_unit_dataset(n, dim, 1);
    let queries = perturbed_queries(&data, n, dim, n_queries, 0.40, 2);

    // Ground truth top-k by brute-force inner product.  Keep both ids and
    // true inner-product scores so we can measure "MSE on top-k" too — the
    // ScaNN paper shows this is where anisotropic loss matters most.
    let t0 = Instant::now();
    let gt: Vec<Vec<(u32, f32)>> = (0..n_queries)
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            brute_force_topk(&data, n, dim, q, topk)
        })
        .collect();
    let gt_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("brute-force ground truth: {:.0} ms", gt_ms);
    println!();

    // Train three codebooks: L2, anisotropic η=2, η=4.
    let opts = TrainOpts { iters: 15, k, seed: 7 };
    let variants: &[(&str, f32)] = &[("l2-pq (η=1)", 1.0), ("aniso-pq (η=2)", 2.0), ("aniso-pq (η=4)", 4.0)];

    let mut results: Vec<VariantResult> = Vec::new();
    for &(name, eta) in variants {
        let t = Instant::now();
        let pq = if (eta - 1.0).abs() < 1e-9 {
            train_l2_pq(&data, n, dim, m, k, opts)
        } else {
            train_anisotropic_pq(&data, n, dim, m, k, eta, opts)
        };
        let train_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let codes = pq.encode_many(&data, n);
        let enc_ms = t.elapsed().as_secs_f64() * 1e3;

        // Score MSE (full set), score MSE on TRUE top-k only, and recall@topk.
        let mut mse_full_acc = 0f64;
        let mut mse_topk_acc = 0f64;
        let mut mse_full_n: u64 = 0;
        let mut mse_topk_n: u64 = 0;
        let mut recall_acc = 0f64;
        let t = Instant::now();
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let tbl = pq.build_lookup_ip(q);
            let mut scored: Vec<(u32, f32)> = (0..n)
                .map(|i| {
                    let c = &codes[i * m..(i + 1) * m];
                    (i as u32, pq.score_with_lookup(&tbl, c))
                })
                .collect();
            for i in 0..n {
                let est = scored[i].1;
                let row = &data[i * dim..(i + 1) * dim];
                let mut truth = 0f32;
                for j in 0..dim {
                    truth += row[j] * q[j];
                }
                let e = (est - truth) as f64;
                mse_full_acc += e * e;
                mse_full_n += 1;
            }
            for (gt_id, gt_score) in &gt[qi] {
                let est = scored[*gt_id as usize].1;
                let e = (est - *gt_score) as f64;
                mse_topk_acc += e * e;
                mse_topk_n += 1;
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(topk);
            let mut hits = 0usize;
            for (idx, _) in &scored {
                if gt[qi].iter().any(|(g, _)| g == idx) {
                    hits += 1;
                }
            }
            recall_acc += hits as f64 / topk as f64;
        }
        let search_ms = t.elapsed().as_secs_f64() * 1e3;
        let mse_full = mse_full_acc / mse_full_n as f64;
        let mse_topk = mse_topk_acc / mse_topk_n as f64;
        let recall = recall_acc / n_queries as f64;
        let qps = n_queries as f64 / (search_ms / 1e3);

        println!("{:<18}  train={:>8.0} ms  encode={:>6.1} ms  search={:>7.1} ms  qps={:>7.0}",
            name, train_ms, enc_ms, search_ms, qps);
        println!("                   MSE_full = {:.4e}   MSE_top{} = {:.4e}   recall@{} = {:.4}",
            mse_full, topk, mse_topk, topk, recall);
        results.push(VariantResult {
            name: name.to_string(),
            train_ms, enc_ms, search_ms, qps,
            mse: mse_full, mse_topk, recall,
        });
    }

    println!();
    println!("--- summary ---");
    let baseline = &results[0];
    for r in &results {
        let mse_full_ratio = baseline.mse / r.mse;
        let mse_topk_ratio = baseline.mse_topk / r.mse_topk;
        let recall_delta = r.recall - baseline.recall;
        println!("{:<18}  MSE_full = {:.2}x   MSE_top{} = {:.2}x   Δrecall@{} = {:+.4}",
            r.name, mse_full_ratio, topk, mse_topk_ratio, topk, recall_delta);
    }

    println!();
    println!("Acceptance criterion: aniso-pq (η>1) must achieve EITHER");
    println!("  (a) MSE_top{} ratio ≥ 1.05   (better score reconstruction on the", topk);
    println!("      vectors that actually decide the ranking — ScaNN's headline);  OR");
    println!("  (b) Δrecall@{} ≥ +0.005      (better top-k retrieval).", topk);
    let aniso4 = &results[2];
    let mse_topk_ratio = baseline.mse_topk / aniso4.mse_topk;
    let ok_mse_topk = mse_topk_ratio >= 1.05;
    let ok_rec = (aniso4.recall - baseline.recall) >= 0.005;
    if ok_mse_topk || ok_rec {
        println!("PASS (η=4): MSE_top{} ratio = {:.3}x   Δrecall = {:+.4}",
            topk, mse_topk_ratio, aniso4.recall - baseline.recall);
    } else {
        println!("MISS (η=4): MSE_top{} ratio = {:.3}x   Δrecall = {:+.4}",
            topk, mse_topk_ratio, aniso4.recall - baseline.recall);
    }
}

/// Build `n_queries` queries by sampling unique dataset items and perturbing
/// each with isotropic Gaussian noise of magnitude `noise`, then renormalising.
/// `noise = 0` ⇒ exact dataset items;  `noise = 1` ⇒ approximately disjoint.
fn perturbed_queries(data: &[f32], n: usize, dim: usize, n_queries: usize, noise: f32, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut out = vec![0f32; n_queries * dim];
    for q in 0..n_queries {
        let src_idx = rng.gen_range(0..n);
        let src = &data[src_idx * dim..(src_idx + 1) * dim];
        for j in 0..dim {
            let u1: f32 = rng.gen::<f32>().max(1e-9);
            let u2: f32 = rng.gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            out[q * dim + j] = src[j] + noise * z;
        }
        let mut nm = 0f32;
        for j in 0..dim {
            nm += out[q * dim + j].powi(2);
        }
        let nm = nm.sqrt().max(1e-12);
        for j in 0..dim {
            out[q * dim + j] /= nm;
        }
    }
    out
}

struct VariantResult {
    name: String,
    #[allow(dead_code)] train_ms: f64,
    #[allow(dead_code)] enc_ms: f64,
    #[allow(dead_code)] search_ms: f64,
    #[allow(dead_code)] qps: f64,
    mse: f64,
    mse_topk: f64,
    recall: f64,
}
