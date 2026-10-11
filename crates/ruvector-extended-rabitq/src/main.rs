//! eRaBitQ demo: build an index, run top-k with each bit width,
//! report real recall vs flat L2 and bytes/vector.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use ruvector_extended_rabitq::{
    error_bound::VarianceBound, quantize::Index, scan::topk_l2, ExtendedRaBitQ,
};
use std::time::Instant;

fn gen_vec(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    // Normalise so IP ~ cosine; keeps numbers readable.
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

fn exact_topk(xs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = xs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let d: f32 = x.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

fn recall(approx: &[(usize, f32)], exact: &[(usize, f32)]) -> f32 {
    let ex: std::collections::HashSet<usize> = exact.iter().map(|p| p.0).collect();
    let hit = approx.iter().filter(|p| ex.contains(&p.0)).count();
    hit as f32 / exact.len() as f32
}

fn main() {
    let dim = 128;
    let n = 5_000;
    let n_queries = 200;
    let k = 10;

    let mut rng = ChaCha8Rng::seed_from_u64(2026_10_11);
    let base: Vec<Vec<f32>> = (0..n).map(|_| gen_vec(&mut rng, dim)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_vec(&mut rng, dim)).collect();

    println!("eRaBitQ demo: dim={} base={} queries={} k={}", dim, n, n_queries, k);
    println!("{:<6} {:<8} {:<10} {:<10} {:<12} {:<10}",
             "bits", "bytes/v", "build_ms", "scan_us/q", "recall@10", "2σ_rel_IP");

    // Exact baseline search is identical across runs; measure once.
    let exact: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| exact_topk(&base, q, k)).collect();

    for bits in [1u8, 2, 3, 4, 6, 8] {
        let q = ExtendedRaBitQ::new(dim, bits, 42).unwrap();
        let padded = q.padded_dim;
        let bytes_per_vec = (padded * bits as usize + 7) / 8 + 4; // + f32 norm
        let bound = VarianceBound::new(bits, padded);
        let t0 = Instant::now();
        let mut idx = Index::new(q);
        idx.add(&base).unwrap();
        let build_ms = t0.elapsed().as_secs_f32() * 1e3;

        let t0 = Instant::now();
        let mut rsum = 0.0f32;
        for (qi, qv) in queries.iter().enumerate() {
            let approx = topk_l2(&idx, qv, k);
            rsum += recall(&approx, &exact[qi]);
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let per_q_us = elapsed / n_queries as f32 * 1e6;
        let r = rsum / n_queries as f32;
        println!(
            "{:<6} {:<8} {:<10.1} {:<10.1} {:<12.3} {:<10.4}",
            bits, bytes_per_vec, build_ms, per_q_us, r, bound.two_sigma_rel_ip()
        );
    }
}
