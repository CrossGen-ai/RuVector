//! Minimal IVF-style scan example comparing full L2, partial-L2 early
//! abandon, and ADSampling on a synthetic gaussian cluster.

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use ruvector_adsampling::{
    topk_adsample, topk_full, topk_partial, Policy, Projector,
};
use std::time::Instant;

fn gauss(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut r = StdRng::seed_from_u64(seed);
    (0..n * d)
        .map(|_| {
            let v: f64 = r.sample(StandardNormal);
            v as f32
        })
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let d: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let k: usize = 10;
    let q_count: usize = 50;

    println!("n={n} d={d} k={k} queries={q_count}");
    let data = gauss(n, d, 42);
    let queries = gauss(q_count, d, 7);
    let p = Projector::new(d, 11);
    let mut pdata = vec![0f32; n * d];
    p.project_many(&data, &mut pdata);

    // --- full L2 baseline ---
    let t = Instant::now();
    let mut truth_sets = Vec::with_capacity(q_count);
    for qi in 0..q_count {
        let q = &queries[qi * d..(qi + 1) * d];
        truth_sets.push(topk_full(&data, d, q, k));
    }
    let full_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "full L2         : {:>7.2} ms total ({:.3} ms/query)",
        full_ms,
        full_ms / q_count as f64
    );

    // --- partial-L2 early abandon ---
    let t = Instant::now();
    let mut partial_hits = 0usize;
    let mut partial_pruned = 0u64;
    let mut partial_full = 0u64;
    for qi in 0..q_count {
        let q = &queries[qi * d..(qi + 1) * d];
        let (res, stats) = topk_partial(&data, d, q, k, 32);
        partial_pruned += stats.pruned;
        partial_full += stats.full;
        let truth = &truth_sets[qi];
        let ids: std::collections::BTreeSet<_> =
            res.iter().map(|(_, i)| *i).collect();
        let tids: std::collections::BTreeSet<_> =
            truth.iter().map(|(_, i)| *i).collect();
        partial_hits += ids.intersection(&tids).count();
    }
    let partial_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "partial-L2      : {:>7.2} ms total ({:.3} ms/query)  recall@{k}={:.3}  prune={:.1}%",
        partial_ms,
        partial_ms / q_count as f64,
        partial_hits as f64 / (q_count * k) as f64,
        100.0 * partial_pruned as f64 / (partial_pruned + partial_full) as f64,
    );

    // --- ADSampling ---
    for eps in [1.5f32, 2.1, 2.6] {
        let pol = Policy::new(32, eps);
        let t = Instant::now();
        let mut hits = 0usize;
        let mut pruned = 0u64;
        let mut full = 0u64;
        for qi in 0..q_count {
            let q = &queries[qi * d..(qi + 1) * d];
            let mut pq = vec![0f32; d];
            p.project(q, &mut pq);
            let (res, stats) = topk_adsample(&pdata, d, &pq, k, pol);
            pruned += stats.pruned;
            full += stats.full;
            let truth = &truth_sets[qi];
            let ids: std::collections::BTreeSet<_> =
                res.iter().map(|(_, i)| *i).collect();
            let tids: std::collections::BTreeSet<_> =
                truth.iter().map(|(_, i)| *i).collect();
            hits += ids.intersection(&tids).count();
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "ADSampling eps={eps:.1}: {:>7.2} ms total ({:.3} ms/query)  recall@{k}={:.3}  prune={:.1}%",
            ms,
            ms / q_count as f64,
            hits as f64 / (q_count * k) as f64,
            100.0 * pruned as f64 / (pruned + full) as f64,
        );
    }
}
