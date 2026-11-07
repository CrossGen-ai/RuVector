//! Real benchmark binary. Builds a small-world layer-0 graph, measures
//! mean edge gap and search latency for three orderings:
//!   * identity
//!   * random
//!   * RCM
//!
//! Prints JSON-like results plus a human-readable table.

use ruvector_rcm_hnsw::*;
use std::time::Instant;

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    use rand::prelude::*;
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn gen_queries(n: usize, dim: usize, seed: u64) -> Vec<Vector> {
    gen_vectors(n, dim, seed)
}

fn run_search_batch(g: &HnswLayer0, entry: NodeId, queries: &[Vector], k: usize, ef: usize) -> (f64, f64) {
    let t = Instant::now();
    let mut visited_total = 0u64;
    for q in queries {
        let (_r, v) = search(g, q, entry, k, ef);
        visited_total += v as u64;
    }
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
    let avg_us = (elapsed_ms * 1000.0) / queries.len() as f64;
    let avg_visited = visited_total as f64 / queries.len() as f64;
    (avg_us, avg_visited)
}

fn recall(gt: &[(NodeId, f32)], got: &[(NodeId, f32)], perm: Option<&[NodeId]>) -> f64 {
    use std::collections::HashSet;
    let gt_set: HashSet<u32> = gt.iter().map(|(i, _)| *i).collect();
    let got_set: HashSet<u32> = got
        .iter()
        .map(|(i, _)| match perm {
            Some(p) => p[*i as usize],
            None => *i,
        })
        .collect();
    gt_set.intersection(&got_set).count() as f64 / gt_set.len() as f64
}

fn brute_force_topk(vectors: &[Vector], q: &[f32], k: usize) -> Vec<(NodeId, f32)> {
    let mut v: Vec<(NodeId, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, x)| (i as NodeId, l2sq(x, q)))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v.truncate(k);
    v
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let sample: usize = std::env::var("SAMPLE").ok().and_then(|s| s.parse().ok()).unwrap_or(96);
    let ef: usize = std::env::var("EF").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let nq: usize = std::env::var("NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);

    eprintln!("[config] n={n} dim={dim} m={m} sample={sample} ef={ef} k={k} nq={nq}");

    eprintln!("[step] generating vectors");
    let vectors = gen_vectors(n, dim, 42);
    let queries = gen_queries(nq, dim, 999);

    eprintln!("[step] building layer-0 small-world graph");
    let g_id = build_smallworld_layer(vectors.clone(), m, sample, 7);

    eprintln!("[step] ground-truth top-k by brute force");
    let gt: Vec<Vec<(NodeId, f32)>> = queries.iter().map(|q| brute_force_topk(&vectors, q, k)).collect();

    // Permutations
    let p_id = perm_identity(g_id.n());
    let p_rnd = perm_random(g_id.n(), 1234);
    let p_rcm = perm_rcm(&g_id.adjacency);
    let inv = |p: &[NodeId]| -> Vec<u32> {
        let mut iv = vec![0u32; p.len()];
        for (new_id, &old) in p.iter().enumerate() { iv[old as usize] = new_id as u32; }
        iv
    };

    let variants = [
        ("identity", g_id.clone(), p_id.clone(), inv(&p_id)),
        ("random", g_id.permute(&p_rnd), p_rnd.clone(), inv(&p_rnd)),
        ("rcm", g_id.permute(&p_rcm), p_rcm.clone(), inv(&p_rcm)),
    ];

    println!("variant,mean_edge_gap,avg_search_us,avg_visited,recall@{k}");
    for (name, g, perm, inv_perm) in &variants {
        let gap = g.mean_edge_gap();
        let entry = inv_perm[0];
        let (us, vis) = run_search_batch(g, entry, &queries, k, ef);

        // Recall
        let mut total_recall = 0.0;
        for (qi, q) in queries.iter().enumerate() {
            let (got, _) = search(g, q, entry, k, ef);
            total_recall += recall(&gt[qi], &got, Some(perm));
        }
        let avg_recall = total_recall / queries.len() as f64;

        println!("{name},{gap:.1},{us:.1},{vis:.1},{avg_recall:.3}");
    }
}
