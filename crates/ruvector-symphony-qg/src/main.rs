//! symphony-qg-demo — end-to-end driver. Prints real recall and timing
//! across the three variants on synthetic Gaussian-mixture data.

use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand::rngs::StdRng;

use ruvector_symphony_qg::{AnnIndex, FlatIndex, PqRerankIndex, SymphonyQgIndex};

fn gaussian_mixture(n: usize, d: usize, k_clusters: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..k_clusters)
        .map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect())
        .collect();
    let mut out = Vec::with_capacity(n * d);
    let normal = Normal::new(0.0, sigma).unwrap();
    for i in 0..n {
        let c = &centers[i % k_clusters];
        for j in 0..d {
            out.push(c[j] + normal.sample(&mut rng) as f32);
        }
    }
    out
}

fn recall_at_k(truth: &[(u32, f32)], got: &[(u32, f32)], k: usize) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().take(k).map(|x| x.0).collect();
    let hit = got.iter().take(k).filter(|x| t.contains(&x.0)).count();
    hit as f32 / k as f32
}

fn main() {
    let n = std::env::var("SYMPHONY_N").ok().and_then(|s| s.parse().ok()).unwrap_or(8_000usize);
    let d = std::env::var("SYMPHONY_D").ok().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    let nq = 200usize;
    let k = 10usize;
    let m_pq = std::env::var("SYMPHONY_M").ok().and_then(|s| s.parse().ok()).unwrap_or(32usize);
    let k_pq = 256usize;
    let m_edges = 32usize;
    let ef_construction = 200usize;
    let ef_search = 200usize;

    println!("=== ruvector-symphony-qg demo ===");
    println!("n={n} d={d} nq={nq} k={k} M={m_pq} K={k_pq} M_edges={m_edges}");
    println!("compression ratio (vs f32): {:.1}x", (d * 4) as f32 / m_pq as f32);

    let base = gaussian_mixture(n, d, 32, 1.2, 17);
    let queries = gaussian_mixture(nq, d, 32, 1.2, 99);

    // Flat (ground truth).
    let flat = FlatIndex::new(d, base.clone());
    let t0 = Instant::now();
    let truths: Vec<Vec<(u32, f32)>> = (0..nq).map(|i| flat.search(&queries[i*d..(i+1)*d], k)).collect();
    let flat_qps = nq as f64 / t0.elapsed().as_secs_f64();
    println!("\n[Flat f32]       qps={:>8.1}", flat_qps);

    // PQ + rerank.
    let rerank_k = 200usize;
    let t = Instant::now();
    let pqr = PqRerankIndex::build(d, base.clone(), m_pq, k_pq, 12, rerank_k, 7);
    let build_pqr = t.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let pqr_res: Vec<Vec<(u32, f32)>> = (0..nq).map(|i| pqr.search(&queries[i*d..(i+1)*d], k)).collect();
    let pqr_qps = nq as f64 / t0.elapsed().as_secs_f64();
    let pqr_recall: f32 = (0..nq).map(|i| recall_at_k(&truths[i], &pqr_res[i], k)).sum::<f32>() / nq as f32;
    println!("[PQ+rerank({rerank_k})] qps={:>8.1}  recall@{k}={:.4}  build={:.2}s", pqr_qps, pqr_recall, build_pqr);

    // Symphony-QG (no rerank).
    let t = Instant::now();
    let sym = SymphonyQgIndex::build(d, base.clone(), m_pq, k_pq, 12, m_edges, ef_construction, 1.2, 7)
        .with_ef_search(ef_search);
    let build_sym = t.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let sym_res: Vec<Vec<(u32, f32)>> = (0..nq).map(|i| sym.search(&queries[i*d..(i+1)*d], k)).collect();
    let sym_qps = nq as f64 / t0.elapsed().as_secs_f64();
    let sym_recall: f32 = (0..nq).map(|i| recall_at_k(&truths[i], &sym_res[i], k)).sum::<f32>() / nq as f32;
    println!("[Symphony-QG]    qps={:>8.1}  recall@{k}={:.4}  build={:.2}s", sym_qps, sym_recall, build_sym);

    // Symphony-QG with small refine for ablation.
    let sym2 = SymphonyQgIndex::build(d, base.clone(), m_pq, k_pq, 12, m_edges, ef_construction, 1.2, 7)
        .with_ef_search(ef_search)
        .with_refine(32);
    let t0 = Instant::now();
    let sym2_res: Vec<Vec<(u32, f32)>> = (0..nq).map(|i| sym2.search(&queries[i*d..(i+1)*d], k)).collect();
    let sym2_qps = nq as f64 / t0.elapsed().as_secs_f64();
    let sym2_recall: f32 = (0..nq).map(|i| recall_at_k(&truths[i], &sym2_res[i], k)).sum::<f32>() / nq as f32;
    println!("[Symphony+ref32] qps={:>8.1}  recall@{k}={:.4}", sym2_qps, sym2_recall);

    println!("\nMemory per vector:");
    println!("  f32:        {} bytes", d * 4);
    println!("  PQ codes:   {} bytes (compression {:.1}x)", m_pq, (d * 4) as f32 / m_pq as f32);
}
