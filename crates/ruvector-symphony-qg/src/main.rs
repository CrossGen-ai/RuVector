//! `symphony-qg-demo` — build a dataset, train all three index variants, and report
//! real benchmark numbers (recall@10, query latency, memory bytes/vector).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use ruvector_symphony_qg::graph::{l2_sq, ExactGraph, GraphParams};
use ruvector_symphony_qg::symphony::{SymphonyQG, SymphonyQGPacked};
use std::time::Instant;

/// Clustered Gaussian dataset (see lib tests). Realistic ANN benchmark surface.
fn make_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_clusters = 64;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect();
            let nrm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for v in c.iter_mut() {
                *v = *v / nrm * 5.0;
            }
            c
        })
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % n_clusters];
        for j in 0..dim {
            let noise: f32 = rng.gen_range(-0.3_f32..0.3);
            out.push(c[j] + noise);
        }
    }
    out
}

fn make_queries(n: usize, dim: usize, dataset_seed: u64, query_seed: u64) -> Vec<f32> {
    // Build queries from the same cluster structure to mirror real workloads.
    let mut rng = ChaCha8Rng::seed_from_u64(query_seed);
    let mut center_rng = ChaCha8Rng::seed_from_u64(dataset_seed);
    let n_clusters = 64;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dim).map(|_| center_rng.gen_range(-1.0_f32..1.0)).collect();
            let nrm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for v in c.iter_mut() {
                *v = *v / nrm * 5.0;
            }
            c
        })
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % n_clusters];
        for j in 0..dim {
            let noise: f32 = rng.gen_range(-0.4_f32..0.4);
            out.push(c[j] + noise);
        }
    }
    out
}

fn brute_force_top_k(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut all: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| {
            let v = &data[i as usize * dim..(i as usize + 1) * dim];
            (i, l2_sq(q, v))
        })
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.into_iter().take(k).map(|x| x.0).collect()
}

fn recall_at_k(ground_truth: &[u32], got: &[u32]) -> f32 {
    let hits = got.iter().filter(|i| ground_truth.contains(i)).count();
    hits as f32 / ground_truth.len() as f32
}

fn main() {
    // Benchmark configuration.
    let n: usize = std::env::var("SYMPHONY_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let dim: usize = std::env::var("SYMPHONY_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let n_queries: usize = std::env::var("SYMPHONY_Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let k: usize = 10;
    let ef_search: usize = 64;
    let rerank_k: usize = 32;

    println!(
        "=== SymphonyQG benchmark ===\n  N={n}, D={dim}, n_queries={n_queries}, k={k}, ef_search={ef_search}, rerank_k={rerank_k}"
    );

    let t0 = Instant::now();
    let data = make_dataset(n, dim, 7);
    let queries = make_queries(n_queries, dim, 7, 99);
    println!("Generated dataset in {:.2?}", t0.elapsed());

    // Ground truth via brute force.
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..n_queries)
        .map(|q| brute_force_top_k(&data, dim, &queries[q * dim..(q + 1) * dim], k))
        .collect();
    println!("Ground truth in {:.2?}", t0.elapsed());

    let params = GraphParams {
        m: 16,
        ef_construction: 96,
        ef_search,
        seed: 0xC0FFEE,
    };

    // ---- Variant A: ExactGraph (exact f32 distances)
    let t0 = Instant::now();
    let g = ExactGraph::build(data.clone(), dim, params);
    let build_a = t0.elapsed();
    let t0 = Instant::now();
    let mut recall_a = 0.0;
    for (q_i, gt_q) in gt.iter().enumerate() {
        let q_vec = &queries[q_i * dim..(q_i + 1) * dim];
        let got = g.search(q_vec, k, ef_search);
        let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        recall_a += recall_at_k(gt_q, &got_ids);
    }
    let elapsed_a = t0.elapsed();
    recall_a /= n_queries as f32;
    let qps_a = n_queries as f64 / elapsed_a.as_secs_f64();
    let mem_a = g.vectors.len() * 4 + g.neighbors.iter().map(|v| v.len() * 4).sum::<usize>();
    println!(
        "A) ExactGraph (baseline): build={:.2?} recall@10={:.4} qps={:.0} latency_us={:.1} mem_per_vec={}B",
        build_a,
        recall_a,
        qps_a,
        elapsed_a.as_secs_f64() * 1e6 / n_queries as f64,
        mem_a / n
    );

    // ---- Variant B: SymphonyQG (rotation + 1-bit codes; re-rank)
    let t0 = Instant::now();
    let s = SymphonyQG::build(data.clone(), dim, params);
    let build_b = t0.elapsed();
    let t0 = Instant::now();
    let mut recall_b = 0.0;
    let mut stats_b_total = 0u64;
    let mut exact_b_total = 0u64;
    for (q_i, gt_q) in gt.iter().enumerate() {
        let q_vec = &queries[q_i * dim..(q_i + 1) * dim];
        let (got, stats) = s.search(q_vec, k, ef_search, rerank_k);
        let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        recall_b += recall_at_k(gt_q, &got_ids);
        stats_b_total += stats.estimated_distance_calls;
        exact_b_total += stats.exact_distance_calls;
    }
    let elapsed_b = t0.elapsed();
    recall_b /= n_queries as f32;
    let qps_b = n_queries as f64 / elapsed_b.as_secs_f64();
    let mem_b = mem_a
        + s.codes.iter().map(|c| c.bits.len() * 8 + 4).sum::<usize>();
    println!(
        "B) SymphonyQG:           build={:.2?} recall@10={:.4} qps={:.0} latency_us={:.1} mem_per_vec={}B est_calls/q={:.0} exact_calls/q={:.0}",
        build_b,
        recall_b,
        qps_b,
        elapsed_b.as_secs_f64() * 1e6 / n_queries as f64,
        mem_b / n,
        stats_b_total as f64 / n_queries as f64,
        exact_b_total as f64 / n_queries as f64,
    );

    // ---- Variant C: SymphonyQGPacked
    let t0 = Instant::now();
    let sp = SymphonyQGPacked::build(data.clone(), dim, params);
    let build_c = t0.elapsed();
    let t0 = Instant::now();
    let mut recall_c = 0.0;
    for (q_i, gt_q) in gt.iter().enumerate() {
        let q_vec = &queries[q_i * dim..(q_i + 1) * dim];
        let (got, _stats) = sp.search(q_vec, k, ef_search, rerank_k);
        let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        recall_c += recall_at_k(gt_q, &got_ids);
    }
    let elapsed_c = t0.elapsed();
    recall_c /= n_queries as f32;
    let qps_c = n_queries as f64 / elapsed_c.as_secs_f64();
    let mem_c = mem_b + sp.packed.iter().map(|b| b.len()).sum::<usize>();
    println!(
        "C) SymphonyQGPacked:     build={:.2?} recall@10={:.4} qps={:.0} latency_us={:.1} mem_per_vec={}B",
        build_c,
        recall_c,
        qps_c,
        elapsed_c.as_secs_f64() * 1e6 / n_queries as f64,
        mem_c / n,
    );

    println!("\nSpeedup B/A = {:.2}x  C/A = {:.2}x  C/B = {:.2}x",
        qps_b / qps_a,
        qps_c / qps_a,
        qps_c / qps_b);
}
