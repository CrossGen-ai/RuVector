//! cache-block-bench — measure adjacency-byte savings, search wall-time, and
//! recall@10 across the three variants on a synthetic 64-d / 10k workload.
//!
//! Numbers are captured into `docs/research/nightly/<date>-cache-block-hnsw/`
//! by the runner; here we just print to stdout.

use ruvector_cache_block_hnsw::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn rand_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect()
}

fn bench<I: AnnIndex>(name: &str, idx: &I, queries: &[Vec<f32>], gt: &[Vec<(NodeId, f32)>], k: usize, ef: usize) {
    // warm
    for q in queries.iter().take(5) { let _ = idx.search(q, k, ef); }
    let t0 = Instant::now();
    let mut hits = 0.0f32;
    for (q, gt_q) in queries.iter().zip(gt.iter()) {
        let got = idx.search(q, k, ef);
        hits += recall_at_k(&got, gt_q, k);
    }
    let dt = t0.elapsed();
    let recall = hits / queries.len() as f32;
    let qps = queries.len() as f64 / dt.as_secs_f64();
    let us_per_q = dt.as_micros() as f64 / queries.len() as f64;
    let bytes = idx.adjacency_bytes();
    println!(
        "{name:<15} adj_bytes={bytes:>10}  recall@{k}={recall:.3}  qps={qps:>8.0}  us/q={us_per_q:>6.1}"
    );
}

fn main() {
    let n = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10_000usize);
    let dim = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let nq = std::env::var("NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(500usize);
    let k = 10usize;
    let ef = 64usize;

    println!("# cache-block-hnsw bench  N={n} dim={dim} nq={nq}  k={k} ef={ef}");
    println!("# host: {}", std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into()));

    println!("building dataset...");
    let vecs = rand_vecs(n, dim, 7);
    let queries = rand_vecs(nq, dim, 77);

    println!("computing ground truth (brute force)...");
    let t = Instant::now();
    let gt: Vec<Vec<(NodeId, f32)>> = queries.iter().map(|q| brute_force_topk(&vecs, q, k)).collect();
    println!("ground truth in {:?}", t.elapsed());

    println!("\nbuilding indexes...");
    let t = Instant::now(); let base = BaselineHnsw::build(&vecs, 16, 64);  println!("  baseline  built in {:?}", t.elapsed());
    let t = Instant::now(); let blk  = BlockHnsw::build(&vecs, 16, 64);     println!("  block     built in {:?}", t.elapsed());
    let t = Instant::now(); let sk_tight = SketchHnsw::build(&vecs, 12, 64).with_slack(20);  println!("  sketch20  built in {:?}", t.elapsed());
    let t = Instant::now(); let sk_safe  = SketchHnsw::build(&vecs, 12, 64).with_slack(64);  println!("  sketch64  built in {:?}", t.elapsed());

    println!();
    bench("baseline",          &base,     &queries, &gt, k, ef);
    bench("block-packed",      &blk,      &queries, &gt, k, ef);
    bench("sketch slack=20",   &sk_tight, &queries, &gt, k, ef);
    bench("sketch slack=64",   &sk_safe,  &queries, &gt, k, ef);
}
