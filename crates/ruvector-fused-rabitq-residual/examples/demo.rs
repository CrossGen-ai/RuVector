//! Runnable benchmark demo:
//! ```
//! cargo run --release -p ruvector-fused-rabitq-residual --bin fused-rq-demo
//! ```
//! Reports code size, per-query latency, and Recall@10 on synthetic
//! Gaussian data for RaBitQ (1-bit), SQ4 (4-bit), and Fused (5-bit).

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use ruvector_fused_rabitq_residual::{
    FusedRQR, QuantizedIndex, Quantizer, RabitQuant, Sq4Quant,
};
use std::time::Instant;

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
                .collect()
        })
        .collect()
}

fn exact_topk(vecs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s = v
                .iter()
                .zip(q.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            (i, s)
        })
        .collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    d.iter().take(k).map(|(i, _)| *i).collect()
}

fn bench<Q: Quantizer>(
    label: &str,
    q: Q,
    vecs: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground: &[Vec<usize>],
    k: usize,
) {
    let code_bytes = q.code_bytes();
    let d = vecs[0].len();
    let bits_per_dim = (code_bytes as f32 * 8.0) / d as f32;

    let t0 = Instant::now();
    let idx = QuantizedIndex::build(q, vecs);
    let build_ms = t0.elapsed().as_secs_f32() * 1000.0;

    // warmup
    let _ = idx.topk(&queries[0], k);

    let t0 = Instant::now();
    let mut recall_sum = 0.0f32;
    for (qi, query) in queries.iter().enumerate() {
        let topk = idx.topk(query, k);
        let got: std::collections::HashSet<usize> =
            topk.iter().map(|(i, _)| *i).collect();
        let want: std::collections::HashSet<usize> =
            ground[qi].iter().take(k).cloned().collect();
        recall_sum += got.intersection(&want).count() as f32 / k as f32;
    }
    let elapsed = t0.elapsed().as_secs_f32();
    let per_query_us = elapsed * 1e6 / queries.len() as f32;
    let recall = recall_sum / queries.len() as f32;

    println!(
        "{:<30} bytes={:>4}  bits/dim={:>4.2}  build={:>7.1}ms  query={:>7.1}µs  recall@{}={:.3}",
        label, code_bytes, bits_per_dim, build_ms, per_query_us, k, recall
    );
}

fn main() {
    let d = 128;
    let n = 8000;
    let m = 200;
    let k = 10;

    println!(
        "Fused RaBitQ+SQ4 benchmark  |  D={d}  N={n}  queries={m}  k={k}  distribution=N(0,I)"
    );
    println!("{:-<110}", "");

    let vecs = gaussian(n, d, 42);
    let queries = gaussian(m, d, 43);
    let t0 = Instant::now();
    let ground: Vec<Vec<usize>> =
        queries.iter().map(|q| exact_topk(&vecs, q, k)).collect();
    println!(
        "ground truth (exact L2): {:.1} ms",
        t0.elapsed().as_secs_f32() * 1000.0
    );
    println!("{:-<110}", "");

    bench("RaBitQ  (1-bit)", RabitQuant::new(d, 7), &vecs, &queries, &ground, k);
    bench("SQ4     (4-bit)", Sq4Quant::new(d, 7), &vecs, &queries, &ground, k);
    bench("Fused   (5-bit)", FusedRQR::new(d, 7), &vecs, &queries, &ground, k);

    println!("{:-<110}", "");
    println!("Notes:");
    println!("  * All quantizers share the same seeded SFHT rotation (fair comparison).");
    println!("  * Fused stores 1-bit signs + 4-bit residual + 3 f32 (norm, res_min, res_step).");
    println!("  * Recall@k is measured against exact-L2 ground truth on raw f32 vectors.");
}
