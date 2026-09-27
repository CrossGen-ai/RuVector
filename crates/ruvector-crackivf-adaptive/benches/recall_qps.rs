//! Harness-free benchmark. Prints real numbers used in the nightly research
//! document — no `criterion`, no mocks.
//!
//! Metrics per variant:
//!   - build time (ms)
//!   - mean search latency (µs) across a fixed skewed workload
//!   - QPS (single thread)
//!   - recall@10
//!   - final nlist
//!   - total splits
//!   - estimated bytes

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_crackivf_adaptive::{
    brute_topk, recall_at_k, AdaptiveThreshold, FixedThreshold, IvfIndex, NoSplit, SplitPolicy,
};
use std::time::Instant;

fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn skewed_queries(vectors: &[Vec<f32>], n_queries: usize, dim: usize, hotspot: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_queries)
        .map(|i| {
            if i % 10 < 7 {
                let base = &vectors[i % hotspot];
                base.iter().map(|x| x + rng.gen::<f32>() * 0.02 - 0.01).collect()
            } else {
                (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
            }
        })
        .collect()
}

/// If `scale_nprobe` is true, the effective `nprobe` grows in proportion to
/// `nlist / nlist_0` on every search. This is how a real CrackIVF deployment
/// keeps recall stable as cells fragment.
fn bench_one<P: SplitPolicy>(
    name: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    nlist_0: usize,
    nprobe: usize,
    policy: P,
    scale_nprobe: bool,
) {
    let t0 = Instant::now();
    let mut idx = IvfIndex::build(vectors.to_vec(), nlist_0, policy, 7).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut recall_sum = 0.0f32;
    let mut lat_ns_total: u128 = 0;
    for q in queries {
        let truth = brute_topk(vectors, q, 10);
        let eff_nprobe = if scale_nprobe {
            let scale = (idx.nlist() as f64 / nlist_0 as f64).ceil() as usize;
            nprobe * scale.max(1)
        } else {
            nprobe
        };
        let t = Instant::now();
        let got = idx.search(q, 10, eff_nprobe).unwrap();
        lat_ns_total += t.elapsed().as_nanos();
        recall_sum += recall_at_k(&got, &truth);
    }
    let recall = recall_sum / queries.len() as f32;
    let mean_lat_us = (lat_ns_total as f64 / queries.len() as f64) / 1e3;
    let qps = 1e6 / mean_lat_us;
    let s = idx.stats();
    println!(
        "{name:<26} build={build_ms:>7.2}ms mean_lat={mean_lat_us:>7.1}µs qps={qps:>8.0} \
recall@10={:.3} nlist_final={} splits={} bytes={}",
        recall,
        idx.nlist(),
        s.splits,
        idx.estimated_bytes()
    );
}

fn main() {
    let n = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(5000usize);
    let dim = std::env::var("D").ok().and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let n_q = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(1000usize);
    let nlist_0 = 16usize;
    let nprobe = 2usize;

    let vectors = gen(n, dim, 1);
    let queries = skewed_queries(&vectors, n_q, dim, 40, 2);

    println!(
        "=== recall_qps: N={n} dim={dim} queries={n_q} nlist_0={nlist_0} nprobe={nprobe} ===",
    );
    println!("-- fixed nprobe (structural: shows latency drop, recall drop) --");
    bench_one("baseline_NoSplit", &vectors, &queries, nlist_0, nprobe, NoSplit, false);
    bench_one(
        "fixed_thresh_p6_s64",
        &vectors,
        &queries,
        nlist_0,
        nprobe,
        FixedThreshold::new(6, 64),
        false,
    );
    bench_one(
        "adaptive_base10_min3",
        &vectors,
        &queries,
        nlist_0,
        nprobe,
        AdaptiveThreshold::new(10, 3, 64),
        false,
    );
    println!("-- scaled nprobe (matched-recall: shows real efficiency win) --");
    bench_one("baseline_NoSplit_scaled", &vectors, &queries, nlist_0, nprobe, NoSplit, true);
    bench_one(
        "fixed_thresh_p6_s64_scaled",
        &vectors,
        &queries,
        nlist_0,
        nprobe,
        FixedThreshold::new(6, 64),
        true,
    );
    bench_one(
        "adaptive_base10_min3_scaled",
        &vectors,
        &queries,
        nlist_0,
        nprobe,
        AdaptiveThreshold::new(10, 3, 64),
        true,
    );
}
