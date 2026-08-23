//! `soar-bench` — real, deterministic benchmark producing numbers that end up
//! in the research doc verbatim. No mocks; no synthetic reporting.
//!
//! Dataset: N=8000 vectors in D=64, gaussian-mixture with 32 latent clusters
//! (well-suited to IVF). Queries: 1000 held-out points drawn from the same
//! mixture. Ground truth: exact brute force. Metric: recall@10.

use std::time::Instant;
use ruvector_soar::{
    brute_force, kmeans, recall_at_k,
    rng::Xor64, IvfNaiveSpill, IvfSoar, IvfTop1, PartitionIndex, Vector,
};

fn synth(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xor64::new(seed);
    // Cluster centers
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(clusters);
    for _ in 0..clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.gauss() * 4.0).collect();
        centers.push(c);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % clusters];
        let v: Vec<f32> = (0..dim).map(|d| c[d] + rng.gauss() * 0.6).collect();
        out.push(v);
    }
    out
}

fn main() {
    let n = 8000usize;
    let d = 64usize;
    let clusters_gen = 32usize;
    let nlist = 128usize;
    let k = 10usize;
    let n_query = 1000usize;

    println!("# ruvector-soar bench");
    println!("dataset: N={n} D={d} clusters_gen={clusters_gen} nlist={nlist} k={k} queries={n_query}");

    let raw = synth(n, d, clusters_gen, 0xC0FFEE);
    let vecs: Vec<Vector> = raw.iter().enumerate()
        .map(|(i, v)| Vector { id: i as u32, data: v.clone() })
        .collect();
    let queries = synth(n_query, d, clusters_gen, 0x8BADF00D);

    // Train shared centroids (deterministic).
    let t0 = Instant::now();
    let centroids = kmeans::train(&raw, nlist, 12, 0x51EED);
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("kmeans_train_ms: {:.2}", train_ms);

    // Ground truth (once, reused).
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = queries.iter()
        .map(|q| brute_force(&vecs, q, k)).collect();
    let gt_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("groundtruth_ms: {:.2}", gt_ms);

    // Build three backends.
    let b0 = Instant::now();
    let ix_top1 = IvfTop1::build(vecs.clone(), centroids.clone());
    let t_top1 = b0.elapsed().as_secs_f64() * 1000.0;

    let b1 = Instant::now();
    let ix_spill = IvfNaiveSpill::build(vecs.clone(), centroids.clone(), 2);
    let t_spill = b1.elapsed().as_secs_f64() * 1000.0;

    let b2 = Instant::now();
    let ix_soar = IvfSoar::build(vecs.clone(), centroids.clone(), 1.5);
    let t_soar = b2.elapsed().as_secs_f64() * 1000.0;

    println!("\n# Build times (ms) and posting sizes");
    println!("{:<16} {:>12} {:>16}", "backend", "build_ms", "posting_bytes");
    for (name, ms, bytes) in [
        (ix_top1.name(), t_top1, ix_top1.posting_bytes()),
        (ix_spill.name(), t_spill, ix_spill.posting_bytes()),
        (ix_soar.name(), t_soar, ix_soar.posting_bytes()),
    ] {
        println!("{:<16} {:>12.2} {:>16}", name, ms, bytes);
    }

    // Estimated bytes = ids in postings × 4 bytes.
    // Baseline (top-1): 8000 * 4 = 32000 bytes.
    // Spill/SOAR:      16000 * 4 = 64000 bytes.
    println!("estimated_posting_bytes: top1={} spill={} soar={}", n*4, n*8, n*8);

    // Sweep nprobe and measure recall + query latency for each backend.
    let sweep = [1usize, 2, 4, 8, 16, 32];
    println!("\n# recall@{k}  and mean query latency (ms)");
    println!("{:<16} {:>8} {:>12} {:>12}", "backend", "nprobe", "recall", "q_ms_mean");
    let backends: Vec<(&str, &dyn PartitionIndex)> = vec![
        (ix_top1.name(), &ix_top1),
        (ix_spill.name(), &ix_spill),
        (ix_soar.name(), &ix_soar),
    ];
    for (name, ix) in &backends {
        for &np in &sweep {
            let mut rsum = 0f32;
            let t0 = Instant::now();
            for (qi, q) in queries.iter().enumerate() {
                let r = ix.search(q, k, np);
                rsum += recall_at_k(&gt[qi], &r);
            }
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let recall = rsum / queries.len() as f32;
            let q_ms = dt_ms / queries.len() as f64;
            println!("{:<16} {:>8} {:>12.4} {:>12.4}", name, np, recall, q_ms);
        }
    }
}
