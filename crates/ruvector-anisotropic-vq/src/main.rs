//! Benchmark: AVQ vs MSE PQ on synthetic unit-norm vectors under MIPS ranking.
//!
//! Reports recall@1, recall@10, and ranking-quality MSE (the mean squared
//! error of the *score estimate* — what actually controls ranking) for
//! several anisotropic eta values and code budgets.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_anisotropic_vq::pq::{ProductQuantizer, QuantizerKind};
use ruvector_anisotropic_vq::search::{brute_force_topk, pq_topk, recall_at_k};
use std::time::Instant;

fn unit_norm(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Standard Gaussian unit-norm corpus — the setting ScaNN was published on.
fn gen_gaussian_unit(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v = vec![0.0f32; d];
            let mut i = 0;
            while i < d {
                let u1: f32 = rng.gen::<f32>().max(1e-9);
                let u2: f32 = rng.gen::<f32>();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                v[i] = r * theta.cos();
                if i + 1 < d {
                    v[i + 1] = r * theta.sin();
                }
                i += 2;
            }
            unit_norm(&mut v);
            v
        })
        .collect()
}

fn encode_all(pq: &ProductQuantizer, data: &[Vec<f32>]) -> Vec<Vec<u8>> {
    data.iter().map(|v| pq.encode(v)).collect()
}

fn ip(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn run_bench(label: &str, n: usize, d: usize, m: usize, k: usize, nq: usize, iters: usize) {
    println!("\n## {label}");
    println!(
        "# N={n}, D={d}, M={m} subquantizers, K={k} codewords/subq, queries={nq}, top-k=10"
    );
    let data = gen_gaussian_unit(n, d, 7);
    let queries = gen_gaussian_unit(nq, d, 99);

    // Ground truth: top-10 by exact IP.
    let gt: Vec<_> = queries.iter().map(|q| brute_force_topk(q, &data, 10)).collect();
    // True IPs for every (q, x) pair — used for the score-estimate MSE.
    let true_ip: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| data.iter().map(|x| ip(q, x)).collect())
        .collect();

    let variants: &[(&str, QuantizerKind)] = &[
        ("MSE (baseline)",      QuantizerKind::Mse),
        ("Aniso eta=1.5",       QuantizerKind::Anisotropic { eta: 1.5 }),
        ("Aniso eta=2.0",       QuantizerKind::Anisotropic { eta: 2.0 }),
        ("Aniso eta=3.0",       QuantizerKind::Anisotropic { eta: 3.0 }),
        ("Aniso eta=4.0",       QuantizerKind::Anisotropic { eta: 4.0 }),
    ];

    println!(
        "\n{:<20} {:>8} {:>10} {:>10} {:>10} {:>14}",
        "variant", "train_ms", "recall@1", "recall@5", "recall@10", "score_MSE×1e3"
    );
    println!("{}", "-".repeat(80));

    for (name, kind) in variants {
        let t = Instant::now();
        let pq = ProductQuantizer::train(d, m, k, *kind, &data, iters, 11);
        let train_ms = t.elapsed().as_secs_f64() * 1000.0;
        let codes = encode_all(&pq, &data);

        let mut r1 = 0.0f32;
        let mut r5 = 0.0f32;
        let mut r10 = 0.0f32;
        let mut score_err2 = 0.0f64;
        let mut count = 0u64;

        for (qi, q) in queries.iter().enumerate() {
            let lut = pq.build_ip_lut(q);
            // recall metrics
            let approx = pq_topk(q, &codes, &pq, 10);
            let gt_top1 = &gt[qi][..1];
            let gt_top5 = &gt[qi][..5.min(gt[qi].len())];
            r1 += recall_at_k(gt_top1, &approx[..1], 1);
            r5 += recall_at_k(gt_top5, &approx[..5.min(approx.len())], 5);
            r10 += recall_at_k(&gt[qi], &approx, 10);
            // score-estimate MSE
            for xi in 0..n {
                let est = pq.score_code(&lut, &codes[xi]);
                let diff = (est - true_ip[qi][xi]) as f64;
                score_err2 += diff * diff;
                count += 1;
            }
        }
        let r1 = r1 / queries.len() as f32 * 100.0;
        let r5 = r5 / queries.len() as f32 * 100.0;
        let r10 = r10 / queries.len() as f32 * 100.0;
        let s_mse = (score_err2 / count as f64) * 1000.0;
        println!(
            "{:<20} {:>8.0} {:>9.1}% {:>9.1}% {:>9.1}% {:>14.3}",
            name, train_ms, r1, r5, r10, s_mse
        );
    }

    let bytes_per_code = m;
    let bytes_per_raw = d * 4;
    let compression = bytes_per_raw as f32 / bytes_per_code as f32;
    println!(
        "# memory: {} B/vec encoded vs {} B/vec raw f32 -> {:.1}x compression",
        bytes_per_code, bytes_per_raw, compression
    );
}

fn main() {
    println!("# Anisotropic Vector Quantization vs MSE PQ");
    println!("# Synthetic Gaussian unit-norm vectors; MIPS top-k ranking");
    println!("# (score_MSE is the average squared error of <q, decode(codes)> vs true <q, x>)");

    // Coarse: K=16 codewords/subq. The regime where loss-aware training
    // helps most because residual energy is large enough to matter.
    run_bench("Coarse codebook (K=16)", 4096, 64, 8, 16, 128, 15);
    // Medium: K=64.
    run_bench("Medium codebook (K=64)", 4096, 64, 8, 64, 128, 12);
    // Production-typical: K=256.
    run_bench("Standard codebook (K=256)", 4096, 64, 8, 256, 128, 10);
}
