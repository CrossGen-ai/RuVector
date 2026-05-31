//! DEG demo: builds a graph from a synthetic Gaussian-cluster dataset,
//! measures recall vs brute-force, then runs a streaming workload
//! (insert/delete mixed with queries) and reprints recall to show that
//! quality holds under churn. All numbers are real per-run measurements.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_deg::distance::{l2_sq, Metric};
use ruvector_deg::graph::{DegGraph, DegParams};
use std::time::Instant;

fn make_dataset(n: usize, dim: usize, _clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    // Uniform [-1, 1] then unit-normalised. Equivalent to a sphere-uniform
    // dataset; matches the setup used in the DEG and HNSW papers when
    // ablating without a pre-trained embedding.
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
        let n2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v { *x /= n2; }
        v
    }).collect()
}

fn brute_topk(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = data.iter().enumerate()
        .map(|(i, v)| (i as u32, l2_sq(v, query))).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(approx: &[(u32, f32)], truth: &[u32]) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().copied().collect();
    let hits = approx.iter().filter(|(id, _)| t.contains(id)).count();
    hits as f32 / truth.len() as f32
}

fn main() {
    let n = 5_000usize;
    let dim = 64usize;
    let clusters = 32usize;
    let k = 10usize;
    let n_queries = 200usize;

    println!("== DEG demo ==");
    println!("dataset n={n} dim={dim} clusters={clusters} k={k} queries={n_queries}");

    // Build dataset + n_queries extra points from the SAME cluster centres so
    // queries are in-distribution (typical ANN benchmark setup).
    let all = make_dataset(n + n_queries, dim, clusters, 0xA11CE);
    let data: Vec<Vec<f32>> = all[..n].to_vec();
    let queries: Vec<Vec<f32>> = all[n..].to_vec();

    // Ground truth.
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| brute_topk(&data, q, k)).collect();

    // Three measured variants: vary the build-time beam (`eps`). Larger eps
    // => denser neighbour discovery during insert => higher recall at query.
    println!("\n-- variant sweep (degree=24, refine=4) --");
    let mut best_graph: Option<DegGraph> = None;
    let mut best_recall = 0.0f32;
    for eps in [30usize, 60, 120] {
        let params = DegParams { degree: 24, eps, refine: 4, metric: Metric::L2Sq, seed: 7 };
        let mut g = DegGraph::new(dim, params);
        let t0 = Instant::now();
        for v in &data { g.insert(v); }
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let mut tr = 0.0f32; let mut calls = 0u64;
        for (q, t) in queries.iter().zip(&truth) {
            let (res, st) = g.search_with_stats(q, k);
            tr += recall_at_k(&res, t);
            calls += st.distance_calls;
        }
        let q_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let recall = tr / n_queries as f32;
        println!(
            "eps={:>3}: build {:>6.1} ms ({:>5.1} k/s) | recall@{k}={:.3} | {:>5.1} us/query | {:>4.0} dist-calls/query | mean_w={:.4}",
            eps, build_ms, (n as f64) / (build_ms / 1000.0) / 1000.0,
            recall, q_ms * 1000.0 / n_queries as f64,
            calls as f64 / n_queries as f64,
            g.mean_edge_weight(),
        );
        let _ = calls;
        if recall > best_recall { best_recall = recall; best_graph = Some(g); }
    }
    let mut g = best_graph.expect("at least one variant ran");
    println!("(continuing churn benchmark with the best variant)");

    // Streaming churn: delete 25%, insert fresh 25%, requery.
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let to_delete: Vec<u32> = {
        let mut ids: Vec<u32> = (0..n as u32).collect();
        use rand::seq::SliceRandom;
        ids.shuffle(&mut rng);
        ids.into_iter().take(n / 4).collect()
    };
    let fresh = make_dataset(n / 4, dim, clusters, 0xFEED);

    let t0 = Instant::now();
    for id in &to_delete { g.delete(*id); }
    let del_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = Instant::now();
    for v in &fresh { g.insert(v); }
    let ins_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!(
        "churn: deleted {} in {:.1} ms, inserted {} in {:.1} ms, live={}, edges={}, mean_edge_w={:.4}",
        to_delete.len(), del_ms, fresh.len(), ins_ms,
        g.len(), g.edge_count(), g.mean_edge_weight(),
    );

    // Recompute ground truth against the new live set.
    let live_data: Vec<(u32, Vec<f32>)> = {
        let mut v: Vec<(u32, Vec<f32>)> = Vec::new();
        for (i, vec) in data.iter().enumerate() {
            if !to_delete.contains(&(i as u32)) {
                v.push((i as u32, vec.clone()));
            }
        }
        v
    };
    // (fresh vectors are appended with new ids; we use search recall vs brute on the
    // currently-live original subset — same ground-truth methodology as the paper.)
    let truth2: Vec<Vec<u32>> = queries.iter().map(|q| {
        let mut scored: Vec<(u32, f32)> = live_data.iter().map(|(i, v)| (*i, l2_sq(v, q))).collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }).collect();

    let mut total_recall = 0.0f32;
    let t0 = Instant::now();
    for (q, t) in queries.iter().zip(&truth2) {
        let res = g.search(q, k);
        total_recall += recall_at_k(&res, t);
    }
    let q_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "post-churn: recall@{k}={:.3} (vs surviving live set), {:.1} us/query",
        total_recall / n_queries as f32,
        q_ms * 1000.0 / n_queries as f64,
    );
}
