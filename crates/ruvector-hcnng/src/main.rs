//! HCNNG demo binary.
//!
//! Runs three measured variants on a synthetic Gaussian-mixture dataset:
//!   1. Brute-force exact L2^2 scan (baseline)
//!   2. HCNNG with n_trees=1 (single random partition tree — ablation)
//!   3. HCNNG with n_trees=12 (paper-recommended ensemble)
//!
//! For each variant we report build_ms, query_us/query, recall@10, and memory.
//! Run with: cargo run --release -p ruvector-hcnng

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use ruvector_hcnng::{HcnngIndex, HcnngParams, Metric};
use std::collections::HashSet;
use std::time::Instant;

fn make_gmm(n: usize, d: usize, clusters: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..d).map(|_| rng.gen_range(-3.0..3.0_f32)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..clusters)];
            c.iter()
                .map(|x| x + rng.gen_range(-sigma..sigma))
                .collect()
        })
        .collect()
}

/// Uniform i.i.d. [-1, 1]^d — used as a "GloVe/SIFT-like" diffuse baseline
/// where data isn't pathologically clumped, so the graph quality dominates
/// the result (not the partition's ability to find the right cluster).
fn make_uniform(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect())
        .collect()
}

fn brute_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(f32, u32)> {
    let mut s: Vec<(f32, u32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut s = 0.0;
            for j in 0..v.len() {
                let dd = v[j] - query[j];
                s += dd * dd;
            }
            (s, i as u32)
        })
        .collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    s.truncate(k);
    s
}

fn recall(got: &[u32], gt: &HashSet<u32>) -> f64 {
    let hit = got.iter().filter(|i| gt.contains(i)).count();
    hit as f64 / got.len().max(1) as f64
}

fn time_ms<F: FnMut() -> R, R>(mut f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let d: usize = std::env::var("D")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let nq: usize = std::env::var("NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let k: usize = 10;

    let dataset = std::env::var("DATASET").unwrap_or_else(|_| "uniform".into());
    println!(
        "HCNNG benchmark — n={}, d={}, queries={}, k={}, dataset={}",
        n, d, nq, k, dataset
    );
    let (data, queries) = match dataset.as_str() {
        "gmm" => {
            println!("Synthetic 32-cluster Gaussian mixture (σ=0.6)");
            (make_gmm(n, d, 32, 0.6, 42), make_gmm(nq, d, 32, 0.6, 99))
        }
        "gmm-soft" => {
            println!("Synthetic 8-cluster Gaussian mixture (σ=1.5)");
            (make_gmm(n, d, 8, 1.5, 42), make_gmm(nq, d, 8, 1.5, 99))
        }
        _ => {
            println!("Uniform i.i.d. on [-1, 1]^d");
            (make_uniform(n, d, 42), make_uniform(nq, d, 99))
        }
    };

    println!("Computing ground truth (brute force) ...");
    let (gt_all, gt_ms): (Vec<HashSet<u32>>, f64) = time_ms(|| {
        queries
            .iter()
            .map(|q| {
                let r = brute_topk(q, &data, k);
                r.into_iter().map(|(_, i)| i).collect()
            })
            .collect()
    });
    let brute_per_q_us = gt_ms * 1000.0 / nq as f64;
    println!(
        "  brute force: {:.1} ms total, {:.1} us/query",
        gt_ms, brute_per_q_us
    );

    println!();
    println!("{:>32}  {:>10}  {:>14}  {:>10}  {:>10}", "variant", "build_ms", "us/query", "recall@10", "edges");
    println!("{}", "-".repeat(82));

    // Variant 1: brute force (already measured above)
    println!(
        "{:>32}  {:>10}  {:>14.2}  {:>10}  {:>10}",
        "brute_force_L2", "-", brute_per_q_us, "1.000", "-"
    );

    // Variant 2: HCNNG n_trees=1
    let params1 = HcnngParams {
        n_trees: 1,
        leaf_size: 32,
        max_degree: 32,
        knn_per_node: 0,
        ef_search: 64,
        seed: 1,
        metric: Metric::L2Sq,
    };
    let (idx1, b1) = time_ms(|| HcnngIndex::build(data.clone(), params1).unwrap());
    let (per_q1_us, rec1) = bench_index(&idx1, &queries, &gt_all, k);
    let e1 = idx1.graph().edge_count();
    println!(
        "{:>32}  {:>10.1}  {:>14.2}  {:>10.3}  {:>10}",
        "hcnng_n_trees=1", b1, per_q1_us, rec1, e1
    );

    // Variant 3: HCNNG n_trees=12 (paper default)
    let params2 = HcnngParams {
        n_trees: 12,
        leaf_size: 32,
        max_degree: 32,
        knn_per_node: 3,
        ef_search: 64,
        seed: 2,
        metric: Metric::L2Sq,
    };
    let (idx2, b2) = time_ms(|| HcnngIndex::build(data.clone(), params2).unwrap());
    let (per_q2_us, rec2) = bench_index(&idx2, &queries, &gt_all, k);
    let e2 = idx2.graph().edge_count();
    println!(
        "{:>32}  {:>10.1}  {:>14.2}  {:>10.3}  {:>10}",
        "hcnng_n_trees=12", b2, per_q2_us, rec2, e2
    );

    // Variant 4: HCNNG n_trees=20 with higher ef
    let params3 = HcnngParams {
        n_trees: 20,
        leaf_size: 32,
        max_degree: 48,
        knn_per_node: 5,
        ef_search: 128,
        seed: 3,
        metric: Metric::L2Sq,
    };
    let (idx3, b3) = time_ms(|| HcnngIndex::build(data.clone(), params3).unwrap());
    let (per_q3_us, rec3) = bench_index(&idx3, &queries, &gt_all, k);
    let e3 = idx3.graph().edge_count();
    println!(
        "{:>32}  {:>10.1}  {:>14.2}  {:>10.3}  {:>10}",
        "hcnng_n_trees=20_ef128", b3, per_q3_us, rec3, e3
    );

    println!();
    println!(
        "Memory: vectors={:.2} MB, graph(n=12)={:.2} MB",
        (idx2.len() * idx2.dim() * 4) as f64 / 1.0e6,
        (idx2.memory_bytes() - idx2.len() * idx2.dim() * 4) as f64 / 1.0e6
    );
    println!(
        "Graph(n_trees=12): avg_degree={:.2}, max_degree={}",
        idx2.graph().avg_degree(),
        idx2.graph().max_degree()
    );

    // Speedup summary
    println!();
    let sp1 = brute_per_q_us / per_q1_us;
    let sp2 = brute_per_q_us / per_q2_us;
    let sp3 = brute_per_q_us / per_q3_us;
    println!("Speedup vs brute force:");
    println!("  n_trees=1   : {:.2}× at recall {:.3}", sp1, rec1);
    println!("  n_trees=12  : {:.2}× at recall {:.3}", sp2, rec2);
    println!("  n_trees=20  : {:.2}× at recall {:.3}", sp3, rec3);

    // Acceptance: best variant (n_trees=20, ef=128) must reach recall@10 >= 0.90
    // on uniform data, which is the standard "competitive ANNS" bar. The
    // n_trees=12 default trades recall for speed — its number is reported but
    // not enforced.
    let pass = rec3 >= 0.90;
    println!();
    println!(
        "ACCEPTANCE: recall@10 (best variant n_trees=20, ef=128) = {:.3}  >= 0.90 ?  {}",
        rec3,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}

fn bench_index(
    idx: &HcnngIndex,
    queries: &[Vec<f32>],
    gt: &[HashSet<u32>],
    k: usize,
) -> (f64, f64) {
    let t = Instant::now();
    let mut tot_r = 0.0;
    for (i, q) in queries.iter().enumerate() {
        let r = idx.search(q, k).unwrap();
        let ids: Vec<u32> = r.iter().map(|s| s.id).collect();
        tot_r += recall(&ids, &gt[i]);
    }
    let elapsed_us = t.elapsed().as_secs_f64() * 1.0e6;
    (elapsed_us / queries.len() as f64, tot_r / queries.len() as f64)
}
