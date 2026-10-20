//! Benchmark harness: measures ms/query, full-distance count and recall for
//! baseline IVF vs pivot-pruned IVF at m=4 and m=8 pivots. All numbers come
//! from this program's own output - captured verbatim in the research doc.

use ruvector_pivot_tri_ivf::{gen_clustered, gen_dataset, IvfIndex, ScalarBackend};
use std::collections::HashSet;
use std::time::Instant;

fn run_variant(
    name: &str,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[HashSet<u32>],
    nlist: usize,
    nprobe: usize,
    k: usize,
    m_pivots: usize,
    use_pivots: bool,
    seed: u64,
) {
    let t0 = Instant::now();
    let idx = IvfIndex::build(data, nlist, m_pivots, 8, seed);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let overhead = idx.pivot_overhead_bytes();

    // Warmup
    for q in queries.iter().take(16) {
        let _ = idx.search(q, k, nprobe, use_pivots);
    }

    let t1 = Instant::now();
    let mut total_full = 0usize;
    let mut total_prune = 0usize;
    let mut total_cand = 0usize;
    let mut hits = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let (res, stats) = idx.search(q, k, nprobe, use_pivots);
        total_full += stats.full_distances;
        total_prune += stats.pivot_prunes;
        total_cand += stats.candidates_seen;
        for h in &res {
            if truth[qi].contains(&h.id) {
                hits += 1;
            }
        }
    }
    let elapsed = t1.elapsed();
    let per_q_ms = (elapsed.as_secs_f64() * 1000.0) / queries.len() as f64;
    let recall = hits as f64 / (queries.len() * k) as f64;
    let prune_pct = if total_cand == 0 {
        0.0
    } else {
        100.0 * total_prune as f64 / total_cand as f64
    };
    println!(
        "{:<24} build={:7.1}ms per_q={:6.3}ms recall={:.4} full_d/q={:7.1} prune/q={:7.1} prune%={:5.2} overhead={}KB",
        name,
        build_ms,
        per_q_ms,
        recall,
        total_full as f64 / queries.len() as f64,
        total_prune as f64 / queries.len() as f64,
        prune_pct,
        overhead / 1024,
    );
}

fn run_dataset(name: &str, data: &[Vec<f32>], queries: &[Vec<f32>], nlist: usize, nprobe: usize, k: usize, seed: u64) {
    let t_bf = Instant::now();
    let truth: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| {
            IvfIndex::<ScalarBackend>::brute_force(data, q, k)
                .into_iter()
                .map(|h| h.id)
                .collect()
        })
        .collect();
    println!("[{}] brute-force truth {:.1}ms", name, t_bf.elapsed().as_secs_f64() * 1000.0);
    run_variant("  baseline(no pivots)", data, queries, &truth, nlist, nprobe, k, 1, false, seed);
    run_variant("  pivot m=4",           data, queries, &truth, nlist, nprobe, k, 4, true,  seed);
    run_variant("  pivot m=8",           data, queries, &truth, nlist, nprobe, k, 8, true,  seed);
}

fn main() {
    let n = 20_000usize;
    let d = 128usize;
    let nq = 200usize;
    let nlist = 128usize;
    let nprobe = 12usize;
    let k = 10usize;
    let seed = 2026;

    println!(
        "ruvector-pivot-tri-ivf bench  n={} d={} nq={} nlist={} nprobe={} k={}",
        n, d, nq, nlist, nprobe, k
    );

    println!();
    println!("Dataset A: iid Gaussian (worst case; concentration of measure)");
    let data = gen_dataset(n, d, seed);
    let queries = gen_dataset(nq, d, seed ^ 0xBEEF);
    run_dataset("iid-gauss", &data, &queries, nlist, nprobe, k, seed);

    println!();
    println!("Dataset B: 32-cluster mixture, sigma=0.6 (realistic embedding topology)");
    let data_c = gen_clustered(n, d, 32, 0.6, seed);
    let queries_c = gen_clustered(nq, d, 32, 0.6, seed ^ 0xBEEF);
    run_dataset("mix-32", &data_c, &queries_c, nlist, nprobe, k, seed);

    println!();
    println!("Dataset C: 64-cluster mixture, sigma=0.3 (tighter blobs)");
    let data_t = gen_clustered(n, d, 64, 0.3, seed);
    let queries_t = gen_clustered(nq, d, 64, 0.3, seed ^ 0xBEEF);
    run_dataset("mix-64-tight", &data_t, &queries_t, nlist, nprobe, k, seed);
}
