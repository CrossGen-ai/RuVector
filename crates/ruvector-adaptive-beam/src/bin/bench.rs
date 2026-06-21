//! Real benchmark for adaptive-beam HNSW.
//!
//! Builds an HNSW over N random vectors, runs Q queries against three
//! terminators, prints recall, mean expansions, and ns/query.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ruvector_adaptive_beam::{
    FixedEfTerminator, Hnsw, HnswParams, QuantileTerminator, RatioTerminator,
};
use std::collections::HashSet;
use std::time::Instant;

fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut v: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, sq_l2(x, q)))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v.iter().take(k).map(|(i, _)| *i).collect()
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|x| x.parse().ok()).unwrap_or(20_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|x| x.parse().ok()).unwrap_or(64);
    let q: usize = std::env::var("Q").ok().and_then(|x| x.parse().ok()).unwrap_or(500);
    let k: usize = 10;
    let ef_max: usize = 256;

    eprintln!("# building HNSW N={n} dim={dim} ef_construction=200 m=16");
    let mut rng = ChaCha8Rng::seed_from_u64(0xA0B0_C0D0);
    let data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();
    let mut idx = Hnsw::new(dim, HnswParams::default());
    let t_build = Instant::now();
    for v in &data {
        idx.insert(v);
    }
    eprintln!("# build ms = {}", t_build.elapsed().as_millis());

    let queries: Vec<Vec<f32>> = (0..q).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();

    // Ground truth.
    eprintln!("# computing ground truth (brute force)…");
    let truths: Vec<Vec<u32>> = queries.iter().map(|qv| brute_topk(&data, qv, k)).collect();

    // ----- Variants -----
    struct Variant {
        name: &'static str,
        run: Box<dyn FnMut(&[Vec<f32>], &Hnsw, &[Vec<u32>]) -> Report>,
    }

    // For an apples-to-apples comparison we sweep efs / ratios / quantiles.
    let mut variants: Vec<Variant> = Vec::new();

    for ef in [32usize, 64, 128] {
        variants.push(Variant {
            name: Box::leak(format!("fixed_ef={ef}").into_boxed_str()),
            run: Box::new(move |qs, ix, gt| {
                let mut term = FixedEfTerminator::new(ef);
                run_variant(qs, ix, gt, &mut term, k)
            }),
        });
    }
    for ratio in [1.05f32, 1.10, 1.20] {
        variants.push(Variant {
            name: Box::leak(format!("ratio={ratio:.2}").into_boxed_str()),
            run: Box::new(move |qs, ix, gt| {
                let mut term = RatioTerminator::new(16, ratio, ef_max);
                run_variant(qs, ix, gt, &mut term, k)
            }),
        });
    }
    for pq in [0.50f64, 0.75, 0.90] {
        variants.push(Variant {
            name: Box::leak(format!("quantile_p={pq:.2}").into_boxed_str()),
            run: Box::new(move |qs, ix, gt| {
                let mut term = QuantileTerminator::new(16, pq, ef_max);
                run_variant(qs, ix, gt, &mut term, k)
            }),
        });
    }

    println!("variant,recall@10,mean_expansions,mean_dist_evals,p50_ns,p95_ns,early_pct");
    for v in &mut variants {
        let r = (v.run)(&queries, &idx, &truths);
        println!(
            "{},{:.4},{:.2},{:.2},{},{},{:.1}",
            v.name, r.recall, r.mean_exp, r.mean_dist, r.p50_ns, r.p95_ns, r.early_pct
        );
    }
}

struct Report {
    recall: f64,
    mean_exp: f64,
    mean_dist: f64,
    p50_ns: u128,
    p95_ns: u128,
    early_pct: f64,
}

fn run_variant<T: ruvector_adaptive_beam::BeamTerminator>(
    queries: &[Vec<f32>],
    idx: &Hnsw,
    truths: &[Vec<u32>],
    term: &mut T,
    k: usize,
) -> Report {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    let mut total = 0usize;
    let mut total_exp = 0u64;
    let mut total_dist = 0u64;
    let mut early = 0u64;
    // Warm-up
    for q in queries.iter().take(20) {
        let _ = idx.search(q, k, term);
    }
    for (q, truth) in queries.iter().zip(truths.iter()) {
        let t0 = Instant::now();
        let (res, stats) = idx.search(q, k, term);
        let ns = t0.elapsed().as_nanos();
        latencies.push(ns);
        let ids: HashSet<u32> = res.iter().map(|n| n.id).collect();
        for t in truth {
            total += 1;
            if ids.contains(t) {
                hits += 1;
            }
        }
        total_exp += stats.expansions;
        total_dist += stats.distance_evals;
        if stats.early_stopped {
            early += 1;
        }
    }
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95) / 100];
    Report {
        recall: hits as f64 / total as f64,
        mean_exp: total_exp as f64 / queries.len() as f64,
        mean_dist: total_dist as f64 / queries.len() as f64,
        p50_ns: p50,
        p95_ns: p95,
        early_pct: 100.0 * early as f64 / queries.len() as f64,
    }
}
