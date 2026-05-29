//! BET-IVF benchmark binary.
//!
//! Builds a synthetic Gaussian-mixture dataset, runs three strategies at
//! matched recall targets, and prints latency / vectors-scored / partitions
//! / recall for each.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_betivf::search::{brute_force_topk, search, SearchStats, SearchStrategy};
use ruvector_betivf::IvfIndex;
use std::collections::HashSet;
use std::time::Instant;

fn make_mixture(dim: usize, n: usize, n_blobs: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(n_blobs);
    let center_noise = Normal::new(0.0, 5.0).unwrap();
    for _ in 0..n_blobs {
        let c: Vec<f32> = (0..dim).map(|_| center_noise.sample(&mut rng) as f32).collect();
        centers.push(c);
    }
    let blob_noise = Normal::new(0.0, 0.5).unwrap();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % n_blobs];
        for j in 0..dim {
            data.push(c[j] + blob_noise.sample(&mut rng) as f32);
        }
    }
    data
}

fn run(
    label: &str,
    idx: &IvfIndex,
    queries: &[f32],
    n_q: usize,
    dim: usize,
    k: usize,
    gt: &[Vec<u32>],
    strategy: SearchStrategy,
) {
    let mut recall_sum = 0.0f32;
    let mut stats_acc = SearchStats::default();
    let mut early = 0usize;
    let t = Instant::now();
    for qi in 0..n_q {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let (res, s) = search(idx, q, k, strategy);
        let res_set: HashSet<u32> = res.iter().map(|(i, _)| *i).collect();
        let gt_set: HashSet<u32> = gt[qi].iter().copied().collect();
        recall_sum += res_set.intersection(&gt_set).count() as f32 / k as f32;
        stats_acc.partitions_scanned += s.partitions_scanned;
        stats_acc.vectors_scored += s.vectors_scored;
        if s.stopped_early {
            early += 1;
        }
    }
    let elapsed = t.elapsed();
    let us = elapsed.as_micros() as f64 / n_q as f64;
    let recall = recall_sum / n_q as f32;
    let scored = stats_acc.vectors_scored as f64 / n_q as f64;
    let parts = stats_acc.partitions_scanned as f64 / n_q as f64;
    println!(
        "{label:<32} recall@{k}={recall:.4}  parts/q={parts:6.2}  vec/q={scored:7.1}  us/q={us:7.1}  early={early}/{n_q}"
    );
}

fn main() {
    let dim = 64;
    let n = 50_000;
    let n_blobs = 256;
    let n_q = 500;
    let k = 10;
    let n_clusters = 256;

    println!("BET-IVF benchmark:  dim={dim}  n={n}  blobs={n_blobs}  clusters={n_clusters}  queries={n_q}  k={k}");
    let data = make_mixture(dim, n, n_blobs, 42);
    let mut rng = StdRng::seed_from_u64(7);
    let t = Instant::now();
    let idx = IvfIndex::build(dim, data, n_clusters, 12, &mut rng).expect("build");
    println!("Built IVF in {:?}", t.elapsed());

    let avg_radius: f32 = idx.partitions.iter().map(|p| p.radius).sum::<f32>() / n_clusters as f32;
    let max_size = idx.partitions.iter().map(|p| p.members.len()).max().unwrap_or(0);
    let min_size = idx.partitions.iter().map(|p| p.members.len()).min().unwrap_or(0);
    println!("Avg radius={avg_radius:.3}  partition size min={min_size} max={max_size}");

    let queries = make_mixture(dim, n_q, n_blobs, 99);
    let t = Instant::now();
    let gt: Vec<Vec<u32>> = (0..n_q)
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            brute_force_topk(&idx, q, k).into_iter().map(|(i, _)| i).collect()
        })
        .collect();
    println!("Computed brute-force ground truth in {:?}", t.elapsed());

    println!("\n--- Strategy comparison ---");
    for np in [4, 8, 16, 32] {
        run(
            &format!("FixedNprobe({np})"),
            &idx,
            &queries,
            n_q,
            dim,
            k,
            &gt,
            SearchStrategy::FixedNprobe(np),
        );
    }
    for b in [400, 800, 1600, 3200] {
        run(
            &format!("FixedBudget({b})"),
            &idx,
            &queries,
            n_q,
            dim,
            k,
            &gt,
            SearchStrategy::FixedBudget(b),
        );
    }
    for slack in [1.0, 1.25, 1.5, 2.0] {
        run(
            &format!("BET(slack={slack})"),
            &idx,
            &queries,
            n_q,
            dim,
            k,
            &gt,
            SearchStrategy::BoundedEarlyTerm {
                max_nprobe: n_clusters,
                slack,
            },
        );
    }
    println!("\nKey: 'slack=1.0' is SOUND (exact stop). slack<1.0 trades recall for latency.");
}
