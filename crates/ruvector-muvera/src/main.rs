//! `cargo run -p ruvector-muvera --release --bin muvera-demo`
//!
//! Demonstrates that FDE inner product approximates ColBERT-style
//! Chamfer similarity, and prints memory/perf numbers for three
//! configurations.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_muvera::{
    chamfer_similarity, FdeConfig, FdeEncoder, FillStrategy, ProjectionMode,
};
use std::time::Instant;

fn unit_multi(n_tokens: usize, d: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    (0..n_tokens)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(rng)).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in v.iter_mut() {
                *x /= n;
            }
            v
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn make_corpus(
    n_docs: usize,
    d_tokens: usize,
    d: usize,
    query: &[Vec<f32>],
    n_planted: usize,
    rng: &mut StdRng,
) -> Vec<Vec<Vec<f32>>> {
    let noise = Normal::new(0.0_f32, 0.05_f32).unwrap();
    let mut docs = Vec::with_capacity(n_docs);
    for i in 0..n_docs {
        let mut doc = unit_multi(d_tokens, d, rng);
        if i < n_planted {
            for qi in 0..query.len().min(d_tokens) {
                let mut v = query[qi].clone();
                for x in v.iter_mut() {
                    *x += noise.sample(rng);
                }
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in v.iter_mut() {
                    *x /= n;
                }
                doc[qi] = v;
            }
        }
        docs.push(doc);
    }
    docs
}

fn run_variant(name: &str, cfg: FdeConfig, n_docs: usize, q_tokens: usize, d_tokens: usize) {
    let d = cfg.d;
    let enc = FdeEncoder::new(cfg.clone()).unwrap();
    let mut rng = StdRng::seed_from_u64(0xC01BE27);
    let query = unit_multi(q_tokens, d, &mut rng);
    let n_planted = 50;
    let docs = make_corpus(n_docs, d_tokens, d, &query, n_planted, &mut rng);

    // exact Chamfer ground truth
    let t0 = Instant::now();
    let exact: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, chamfer_similarity(&query, doc)))
        .collect();
    let chamfer_ms = t0.elapsed().as_secs_f64() * 1e3;

    // FDE encode + score
    let t1 = Instant::now();
    let q_fde = enc.encode_query(&query).unwrap();
    let q_enc_us = t1.elapsed().as_micros();

    let t2 = Instant::now();
    let d_fdes: Vec<Vec<f32>> = docs.iter().map(|doc| enc.encode_doc(doc).unwrap()).collect();
    let d_enc_ms = t2.elapsed().as_secs_f64() * 1e3;

    let t3 = Instant::now();
    let approx: Vec<(usize, f32)> = d_fdes
        .iter()
        .enumerate()
        .map(|(i, df)| (i, dot(&q_fde, df)))
        .collect();
    let score_ms = t3.elapsed().as_secs_f64() * 1e3;

    let mut exact_sorted = exact.clone();
    exact_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let k = n_planted;
    let true_top: std::collections::HashSet<usize> =
        exact_sorted.iter().take(k).map(|x| x.0).collect();
    let mut approx_sorted = approx.clone();
    approx_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let approx_top: std::collections::HashSet<usize> =
        approx_sorted.iter().take(k).map(|x| x.0).collect();
    let recall10 = true_top.intersection(&approx_top).count() as f32 / k as f32;

    let bytes = enc.bytes_per_vector();
    let multi_bytes = d_tokens * d * 4;
    let compression = multi_bytes as f32 / bytes as f32;

    println!("--- {name} ---");
    println!(
        "  cfg: d={} k_sim={} R={} fill={:?} proj={:?} d_final={}",
        cfg.d, cfg.k_sim, cfg.r_reps, cfg.fill, cfg.projection, cfg.d_final
    );
    println!("  output dim:           {} f32 ({} bytes/doc)", enc.config().output_dim(), bytes);
    println!("  raw multi-vec bytes:  {} (compression vs single-vec FDE: {:.2}x)", multi_bytes, compression);
    println!("  recall@{k} vs Chamfer: {:.2}", recall10);
    println!("  query encode:         {} us", q_enc_us);
    println!("  doc   encode (n={n_docs}): {:.2} ms ({:.1} us/doc)", d_enc_ms, d_enc_ms * 1e3 / n_docs as f64);
    println!("  FDE score (n={n_docs}):    {:.2} ms ({:.1} us/doc)", score_ms, score_ms * 1e3 / n_docs as f64);
    println!("  exact Chamfer:        {:.2} ms ({:.1} us/doc)", chamfer_ms, chamfer_ms * 1e3 / n_docs as f64);
    println!("  speedup (scoring):    {:.1}x", chamfer_ms / score_ms.max(1e-6));
}

fn main() {
    let d = 64;
    let n_docs = 1000;
    let q_tokens = 16;
    let d_tokens = 32;

    println!("MUVERA-FDE demo — d={d} q_tokens={q_tokens} d_tokens={d_tokens} n_docs={n_docs}\n");

    run_variant(
        "v1 baseline (k=4, R=4, no proj, zero fill)",
        FdeConfig {
            d, k_sim: 4, r_reps: 4,
            fill: FillStrategy::Zero,
            projection: ProjectionMode::None, d_final: 0, seed: 1,
        },
        n_docs, q_tokens, d_tokens,
    );
    run_variant(
        "v2 paper-style (k=5, R=20, no proj, nearest-bucket fill)",
        FdeConfig {
            d, k_sim: 5, r_reps: 20,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::None, d_final: 0, seed: 2,
        },
        n_docs, q_tokens, d_tokens,
    );
    run_variant(
        "v3 compressed (k=5, R=20, Gaussian proj -> 1024)",
        FdeConfig {
            d, k_sim: 5, r_reps: 20,
            fill: FillStrategy::NearestBucket,
            projection: ProjectionMode::Gaussian, d_final: 1024, seed: 3,
        },
        n_docs, q_tokens, d_tokens,
    );
}
