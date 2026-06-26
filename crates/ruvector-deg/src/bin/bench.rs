//! Benchmark binary that produces real numbers consumed by the research doc.
//!
//! Outputs build time, query time, recall@10, edge count, and bytes per node
//! for the three backends across one corpus size.
//!
//! Run: `cargo run --release -p ruvector-deg --bin deg-bench`

use rand::prelude::*;
use std::time::Instant;
use ruvector_deg::{AnnIndex, DegConfig, DegGraph, NswGraph, RandomGraph, metric::sq_l2};

fn brute(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut s: Vec<(u32, f32)> = data.iter().enumerate()
        .map(|(i, v)| (i as u32, sq_l2(q, v))).collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn run<I: AnnIndex>(
    name: &str,
    mut idx: I,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    gold: &[Vec<u32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f32, usize) {
    let t0 = Instant::now();
    for v in data { idx.insert(v.clone()); }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut hits = 0usize;
    let t0 = Instant::now();
    for (q, g) in queries.iter().zip(gold.iter()) {
        let r = idx.search(q, k, ef);
        let set: std::collections::HashSet<u32> = g.iter().copied().collect();
        hits += r.iter().filter(|(i, _)| set.contains(i)).count();
    }
    let query_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let recall = hits as f32 / (queries.len() * k) as f32;
    let edges = idx.edge_count();
    println!("{name:>8}  build={build_ms:8.1} ms  query={query_ms:7.2} ms  recall@{k}={recall:.3}  edges={edges}");
    (build_ms, query_ms, recall, edges)
}

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let dim: usize = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let nq: usize = std::env::var("NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let k: usize = 10;
    let ef: usize = 64;
    let degree: usize = 16;

    println!("ruvector-deg benchmark");
    println!("N={n} dim={dim} nq={nq} k={k} ef={ef} degree={degree}");

    let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
    let data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()).collect();
    let queries: Vec<Vec<f32>> = (0..nq).map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()).collect();
    let gold: Vec<Vec<u32>> = queries.iter().map(|q| brute(&data, q, k)).collect();

    let rg = RandomGraph::new(dim, degree, 7);
    let (b1, q1, r1, e1) = run("random", rg, &data, &queries, &gold, k, ef);

    let ns = NswGraph::new(dim, degree, 64);
    let (b2, q2, r2, e2) = run("nsw", ns, &data, &queries, &gold, k, ef);

    let mut cfg = DegConfig::default();
    cfg.degree = degree;
    cfg.ef_construction = 64;
    cfg.optimize_every = 256;
    cfg.optimize_passes = 1;
    let mut dg = DegGraph::new(dim, cfg);
    let t0 = Instant::now();
    for v in &data { dg.insert(v.clone()); }
    let _final = dg.optimize_all(2);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut hits = 0usize;
    let t0 = Instant::now();
    for (q, g) in queries.iter().zip(gold.iter()) {
        let r = dg.search(q, k, ef);
        let set: std::collections::HashSet<u32> = g.iter().copied().collect();
        hits += r.iter().filter(|(i, _)| set.contains(i)).count();
    }
    let query_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let recall = hits as f32 / (queries.len() * k) as f32;
    let edges = dg.edge_count();
    println!("{:>8}  build={:8.1} ms  query={:7.2} ms  recall@{}={:.3}  edges={}",
             "deg", build_ms, query_ms, k, recall, edges);

    let bytes_per_node = dim * 4 + degree * 4; // vec + edges
    println!("\nbytes/node ≈ {} ({} dim*f32 + {} degree*u32)", bytes_per_node, dim * 4, degree * 4);
    println!("memory @ N={}: ≈ {:.1} MiB", n, (n * bytes_per_node) as f64 / 1048576.0);

    println!("\nsummary CSV:");
    println!("variant,build_ms,query_ms,recall,edges");
    println!("random,{b1:.1},{q1:.2},{r1:.3},{e1}");
    println!("nsw,{b2:.1},{q2:.2},{r2:.3},{e2}");
    println!("deg,{build_ms:.1},{query_ms:.2},{recall:.3},{edges}");
}
