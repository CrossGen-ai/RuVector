//! Real benchmark binary — generates synthetic unit-sphere data, builds IVF
//! centroids, and measures recall@nprobe + mean rank latency for the three
//! probe rankers. The stdout is captured verbatim by the research README.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use ruvector_cpolytope_lsh_ivf::{
    CosineRanker, CrossPolytopeRanker, L2Ranker, ProbeRanker,
};
use std::time::Instant;

const N: usize = 50_000;
const DIM: usize = 128;
const K: usize = 1_024;
const Q: usize = 1_000;
const TOP_K: usize = 10;
const NPROBES: &[usize] = &[1, 4, 16, 64];
const L_HASHES: usize = 16;
const CENTROID_SAMPLE_SEED: u64 = 0xC0FFEE_u64;
const DATA_SEED: u64 = 0xA1_u64;
const QUERY_SEED: u64 = 0xB2_u64;
const HASH_SEED: u64 = 0xD3_u64;

fn sample_unit_sphere(dim: usize, rng: &mut ChaCha12Rng) -> Vec<f32> {
    // Standard Marsaglia trick: Gaussian sample then normalize.
    // Box–Muller from two uniforms.
    let mut v = Vec::with_capacity(dim);
    while v.len() < dim {
        let u1: f32 = rng.gen_range(1e-9f32..1.0);
        let u2: f32 = rng.gen_range(0.0f32..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        v.push(r * theta.cos());
        if v.len() < dim {
            v.push(r * theta.sin());
        }
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

fn build_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    (0..n).map(|_| sample_unit_sphere(dim, &mut rng)).collect()
}

fn pick_centroids(data: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    // Random sampling from the dataset. The task spec explicitly allows this
    // as an acceptable IVF partitioner stand-in; it keeps the benchmark
    // honest because the same centroids are shared by all three rankers.
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut chosen = Vec::with_capacity(k);
    let mut seen = std::collections::HashSet::new();
    while chosen.len() < k {
        let i: usize = rng.gen_range(0..data.len());
        if seen.insert(i) {
            chosen.push(data[i].clone());
        }
    }
    chosen
}

fn assign_to_nearest(data: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    // Dot-product assignment (unit sphere, cosine).
    data.iter()
        .map(|v| {
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for (ci, c) in centroids.iter().enumerate() {
                let mut s = 0.0f32;
                for (a, b) in v.iter().zip(c.iter()) { s += a * b; }
                if s > best_s { best_s = s; best = ci; }
            }
            best
        })
        .collect()
}

fn exact_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut s = 0.0f32;
            for (a, b) in query.iter().zip(v.iter()) { s += a * b; }
            (i, s)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_nprobe(
    truth: &[usize],
    probe_order: &[(usize, f32)],
    assignments: &[usize],
    data: &[Vec<f32>],
    query: &[f32],
    nprobe: usize,
    top_k: usize,
) -> f32 {
    // Collect the union of posting lists for the top nprobe cells, then
    // exact-rescan them to produce top-k, and compare to truth.
    use std::collections::HashSet;
    let probed_cells: HashSet<usize> =
        probe_order.iter().take(nprobe).map(|(c, _)| *c).collect();
    let mut cand: Vec<(usize, f32)> = assignments
        .iter()
        .enumerate()
        .filter(|(_, c)| probed_cells.contains(c))
        .map(|(vid, _)| {
            let mut s = 0.0f32;
            for (a, b) in query.iter().zip(data[vid].iter()) { s += a * b; }
            (vid, s)
        })
        .collect();
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cand_top: Vec<usize> = cand.into_iter().take(top_k).map(|(i, _)| i).collect();
    let truth_set: std::collections::HashSet<usize> = truth.iter().copied().collect();
    let hit: usize = cand_top.iter().filter(|i| truth_set.contains(i)).count();
    hit as f32 / top_k as f32
}

fn mean(xs: &[f32]) -> f32 { xs.iter().sum::<f32>() / xs.len() as f32 }

fn time_ranker_avg_us(
    ranker: &dyn ProbeRanker,
    queries: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> (f32, Vec<Vec<(usize, f32)>>) {
    // One warm-up query to touch caches.
    let _ = ranker.rank(&queries[0], centroids);
    let mut orders = Vec::with_capacity(queries.len());
    let t0 = Instant::now();
    for q in queries {
        let r = ranker.rank(q, centroids);
        orders.push(r);
    }
    let elapsed = t0.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1e6 / queries.len() as f64;
    (avg_us as f32, orders)
}

fn main() {
    println!("ruvector-cpolytope-lsh-ivf bench");
    println!("  N={}  dim={}  K={}  Q={}  top_k={}  L_hashes={}", N, DIM, K, Q, TOP_K, L_HASHES);
    println!("  seeds: data={:#x} centroids={:#x} queries={:#x} hashes={:#x}",
             DATA_SEED, CENTROID_SAMPLE_SEED, QUERY_SEED, HASH_SEED);

    let t0 = Instant::now();
    let data = build_dataset(N, DIM, DATA_SEED);
    let centroids = pick_centroids(&data, K, CENTROID_SAMPLE_SEED);
    let assignments = assign_to_nearest(&data, &centroids);
    println!("  dataset + IVF partition built in {:.2}s", t0.elapsed().as_secs_f32());

    let mut qrng = ChaCha12Rng::seed_from_u64(QUERY_SEED);
    let queries: Vec<Vec<f32>> = (0..Q).map(|_| sample_unit_sphere(DIM, &mut qrng)).collect();

    let t0 = Instant::now();
    let truths: Vec<Vec<usize>> = queries.iter().map(|q| exact_topk(q, &data, TOP_K)).collect();
    println!("  truth top-{} computed in {:.2}s", TOP_K, t0.elapsed().as_secs_f32());

    let l2 = L2Ranker;
    let cos = CosineRanker::new(&centroids);
    let cp = CrossPolytopeRanker::new(&centroids, L_HASHES, HASH_SEED);

    println!();
    println!(
        "  cross-polytope ranker memory overhead: {} bytes / centroid (L={}, pad_dim={})",
        cp.memory_bytes_per_centroid(),
        cp.l_hashes(),
        cp.pad_dim()
    );

    let rankers: Vec<&dyn ProbeRanker> = vec![&l2, &cos, &cp];

    println!();
    println!("| ranker         | avg rank µs | recall@1 | recall@4 | recall@16 | recall@64 |");
    println!("|----------------|------------:|---------:|---------:|----------:|----------:|");

    for r in rankers {
        let (avg_us, orders) = time_ranker_avg_us(r, &queries, &centroids);
        let mut recalls = Vec::with_capacity(NPROBES.len());
        for &np in NPROBES {
            let per_query: Vec<f32> = (0..queries.len())
                .map(|qi| recall_at_nprobe(
                    &truths[qi], &orders[qi], &assignments, &data, &queries[qi], np, TOP_K,
                ))
                .collect();
            recalls.push(mean(&per_query));
        }
        println!(
            "| {:<14} | {:>11.2} | {:>8.4} | {:>8.4} | {:>9.4} | {:>9.4} |",
            r.name(), avg_us, recalls[0], recalls[1], recalls[2], recalls[3]
        );
    }

    println!();
    println!("done.");
}
