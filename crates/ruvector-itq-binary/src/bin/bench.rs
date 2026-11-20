//! Real end-to-end benchmark for the three rotation backends.
//!
//! Metrics printed:
//!   * training MSE of the binary quantisation
//!   * encode throughput (vectors/sec)
//!   * recall@10 of Hamming top-k against float-L2 ground truth
//!   * query throughput (queries/sec) of the Hamming index

use std::time::Instant;

use ruvector_itq_binary::{
    bruteforce_topk_l2, recall_at_k, BinaryIndex, ItqEncoder, RotationKind, SplitMix64,
};

/// Anisotropic Gaussian — the setting where PCA+ITQ is designed to pay off.
/// Spectrum is a power-law σ_j = (1 + j)^(-0.6) so the top directions dominate
/// variance. This mimics real embedding geometry (SIFT/GIST-like).
fn aniso_gauss(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut scales = vec![0f32; dim];
    for j in 0..dim { scales[j] = (1.0 + j as f32).powf(-0.6); }
    // random orthogonal basis Q (dim x dim) — rotate the diagonal covariance.
    let mut q = vec![0f32; dim * dim];
    for v in &mut q { *v = rng.next_normal() as f32; }
    // Modified Gram-Schmidt
    for j in 0..dim {
        for i in 0..j {
            let mut dot = 0.0f32;
            for r in 0..dim { dot += q[r * dim + j] * q[r * dim + i]; }
            for r in 0..dim { q[r * dim + j] -= dot * q[r * dim + i]; }
        }
        let mut nrm = 0.0f32;
        for r in 0..dim { nrm += q[r * dim + j] * q[r * dim + j]; }
        let nrm = nrm.sqrt().max(1e-20);
        for r in 0..dim { q[r * dim + j] /= nrm; }
    }
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        // z = Q diag(scales) · g  where g ~ N(0,I)
        let g: Vec<f32> = (0..dim).map(|_| rng.next_normal() as f32).collect();
        for r in 0..dim {
            let mut s = 0.0f32;
            for j in 0..dim { s += q[r * dim + j] * scales[j] * g[j]; }
            out.push(s);
        }
    }
    out
}

fn main() {
    // Dataset config — kept modest so the whole run finishes in <~60s on a laptop.
    let n_train: usize = 10_000;
    let n_query: usize = 500;
    let dim: usize = 128;
    let bits_set: &[usize] = &[32, 64, 128];
    let k: usize = 10;
    let r_shortlist: usize = 100; // candidate shortlist size for recall@R
    let seed: u64 = 2026_11_20;

    println!("# ruvector-itq-binary bench");
    println!("# dataset: anisotropic Gaussian (power-law covariance, σ_j = (1+j)^-0.6)");
    println!(
        "# n_train={n_train} n_query={n_query} dim={dim} k={k} shortlist={r_shortlist} seed={seed}"
    );
    println!();

    let data = aniso_gauss(n_train, dim, seed);
    let queries = aniso_gauss(n_query, dim, seed ^ 0x55);

    // Ground truth float L2 top-k for each query.
    let t_gt = Instant::now();
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(n_query);
    for q in queries.chunks_exact(dim) {
        truth.push(bruteforce_topk_l2(&data, dim, q, k));
    }
    let gt_s = t_gt.elapsed().as_secs_f32();
    println!("ground-truth L2 top-{k}: {:.2}s", gt_s);
    println!();

    println!(
        "{:<8} {:<10} {:>10} {:>12} {:>12} {:>10} {:>12}",
        "bits", "rotation", "train_mse", "enc_vecs/s", "query_q/s", "r@10", "r@10in100"
    );
    println!("{}", "-".repeat(84));

    for &bits in bits_set {
        for &kind in &[RotationKind::Identity, RotationKind::Random, RotationKind::Itq] {
            let n_iter = if matches!(kind, RotationKind::Itq) { 50 } else { 0 };
            let t_fit = Instant::now();
            let enc = ItqEncoder::fit(&data, dim, bits, kind, n_iter, seed);
            let fit_s = t_fit.elapsed().as_secs_f32();

            let mse = enc.training_mse(&data);

            // Encode throughput — re-encode database.
            let t_enc = Instant::now();
            let codes = enc.encode_many(&data);
            let enc_s = t_enc.elapsed().as_secs_f32();
            let enc_rate = n_train as f32 / enc_s;

            let idx = BinaryIndex::new(bits, codes);

            // Query loop — both strict top-k recall and shortlist recall.
            let mut sum_r10 = 0.0f32;
            let mut sum_shortlist = 0.0f32;
            let t_q = Instant::now();
            for (qi, q) in queries.chunks_exact(dim).enumerate() {
                let qc = enc.encode(q);
                let hits_short = idx.search(&qc, r_shortlist);
                let pred_short: Vec<u32> = hits_short.iter().map(|&(i, _)| i).collect();
                let pred_top: Vec<u32> = pred_short.iter().take(k).copied().collect();
                sum_r10 += recall_at_k(&pred_top, &truth[qi]);
                sum_shortlist += recall_at_k(&pred_short, &truth[qi]);
            }
            let q_s = t_q.elapsed().as_secs_f32();
            let q_rate = n_query as f32 / q_s;
            let r10 = sum_r10 / n_query as f32;
            let r_short = sum_shortlist / n_query as f32;

            println!(
                "{:<8} {:<10} {:>10.4} {:>12.0} {:>12.0} {:>10.4} {:>12.4}  (fit {:.2}s)",
                bits,
                match kind {
                    RotationKind::Identity => "none",
                    RotationKind::Random => "random",
                    RotationKind::Itq => "itq",
                },
                mse,
                enc_rate,
                q_rate,
                r10,
                r_short,
                fit_s,
            );
        }
        println!();
    }
}
