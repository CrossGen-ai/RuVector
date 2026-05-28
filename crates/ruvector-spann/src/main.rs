//! Demo + benchmark binary: build three variants, measure recall@10, latency,
//! memory, and replication factor on a synthetic clustered dataset.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_spann::{
    FixedMultiAssign, PolicyKind, SingleAssign, SpannClosure, SpannIndex,
};
use ruvector_spann::policy::ClosurePolicy;

/// Clustered Gaussian-mixture data — closer to real-world structure than uniform.
fn gen_clustered(n: usize, d: usize, n_clusters: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| rng.gen::<f32>() * 10.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..n_clusters)];
            (0..d).map(|i| c[i] + (rng.gen::<f32>() - 0.5) * 2.0 * sigma).collect()
        })
        .collect()
}

fn run_variant(
    name: &str,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k_centroids: usize,
    policy: &dyn ClosurePolicy,
    nprobe: usize,
    top_k: usize,
) {
    let t0 = Instant::now();
    let (idx, stats) = SpannIndex::build(
        data.to_vec(),
        k_centroids,
        15,
        policy,
        42,
    );
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut hits = 0usize;
    let mut total = 0usize;
    for (q, gt) in queries.iter().zip(ground_truth.iter()) {
        let res = idx.search(q, top_k, nprobe);
        for r in &res {
            if gt.contains(&r.id) {
                hits += 1;
            }
        }
        total += top_k;
    }
    let search_us_per_query =
        t1.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    let recall = hits as f32 / total as f32;

    let mem_mb = idx.mem_bytes() as f64 / (1024.0 * 1024.0);

    println!(
        "{:<20} kind={:?}  rep={:>5.2}  recall@{}={:.4}  search={:>6.1}µs  build={:>7.1}ms  mem={:>5.2}MB  postings={}",
        name,
        policy.kind(),
        stats.replication_factor,
        top_k,
        recall,
        search_us_per_query,
        build_ms,
        mem_mb,
        stats.n_postings_entries,
    );

    let _ = PolicyKind::Single; // silence unused if compiler decides.
}

fn ground_truth(data: &[Vec<f32>], queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<usize>> {
    // Build a throwaway single-assign index just to reuse brute_force.
    let (idx, _) = SpannIndex::build(
        data.to_vec(),
        1,
        1,
        &SingleAssign,
        0,
    );
    queries
        .iter()
        .map(|q| idx.brute_force(q, top_k).into_iter().map(|r| r.id).collect())
        .collect()
}

fn main() {
    let n = std::env::var("SPANN_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let dim = std::env::var("SPANN_D").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let nq = 200;
    let k_centroids = 128;
    let top_k = 10;

    println!("== ruvector-spann demo ==");
    println!(
        "n={}  dim={}  queries={}  K(centroids)={}  top_k={}",
        n, dim, nq, k_centroids, top_k
    );

    let data = gen_clustered(n, dim, 64, 1.0, 1);
    let queries = gen_clustered(nq, dim, 64, 1.0, 2);

    let t = Instant::now();
    let gt = ground_truth(&data, &queries, top_k);
    println!("ground truth (brute force): {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

    for &nprobe in &[2usize, 4, 8, 16] {
        println!("\n--- nprobe = {} ---", nprobe);
        run_variant("baseline-single", &data, &queries, &gt, k_centroids, &SingleAssign, nprobe, top_k);
        run_variant(
            "fixed-multi(k=2)",
            &data, &queries, &gt, k_centroids,
            &FixedMultiAssign { k: 2 }, nprobe, top_k,
        );
        run_variant(
            "fixed-multi(k=4)",
            &data, &queries, &gt, k_centroids,
            &FixedMultiAssign { k: 4 }, nprobe, top_k,
        );
        run_variant(
            "spann(eps=0.10,cap=4)",
            &data, &queries, &gt, k_centroids,
            &SpannClosure { epsilon: 0.10, cap: 4 }, nprobe, top_k,
        );
        run_variant(
            "spann(eps=0.20,cap=8)",
            &data, &queries, &gt, k_centroids,
            &SpannClosure { epsilon: 0.20, cap: 8 }, nprobe, top_k,
        );
    }
}
