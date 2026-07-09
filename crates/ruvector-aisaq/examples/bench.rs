//! Real benchmark for ruvector-aisaq.
//!
//! Produces numbers used in the research doc and the public gist.
//! No mocks, no aspirational figures — every row comes from
//! `Instant::now()` around real search calls over synthetic data.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_aisaq::backends::{DistanceBackend, FlatF32Ram, PqDisk, PqRam};
use ruvector_aisaq::graph::{BeamSearcher, KnnGraph};
use ruvector_aisaq::pq::ProductQuantizer;
use ruvector_aisaq::{l2_sq, recall_at_k};

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0f32; n * d];
    for v in out.iter_mut() {
        // Box-Muller-lite: two uniforms -> approximately normal
        let u1: f32 = rng.gen_range(1e-9..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    out
}

fn brute_force_topk(vectors: &[f32], d: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = vectors.len() / d;
    let mut dists: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, l2_sq(query, &vectors[i * d..(i + 1) * d])))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(i, _)| i).collect()
}

fn run_variant<B: DistanceBackend>(
    label: &str,
    graph: &KnnGraph,
    backend: B,
    queries: &[f32],
    d: usize,
    truth: &[Vec<u32>],
    k: usize,
    beam: usize,
) -> (String, usize, f64, f32) {
    let ram = backend.ram_bytes();
    let mut searcher = BeamSearcher::new(graph, backend, beam);
    let nq = queries.len() / d;

    let t0 = Instant::now();
    let mut recalls = 0.0f32;
    for q in 0..nq {
        let query = &queries[q * d..(q + 1) * d];
        let got = searcher.search(query, k);
        recalls += recall_at_k(&got, &truth[q]);
    }
    let elapsed = t0.elapsed();
    let per_query_us = elapsed.as_secs_f64() * 1e6 / nq as f64;
    let recall = recalls / nq as f32;
    (label.to_string(), ram, per_query_us, recall)
}

fn main() {
    let n: usize = std::env::var("AISAQ_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let d: usize = 128;
    let m: usize = 16;    // 16 subquantizers -> 16-byte codes
    let r: usize = 32;    // graph degree
    let beam: usize = 64; // search beam
    let k: usize = 10;
    let nq: usize = 200;

    println!("# ruvector-aisaq benchmark");
    println!("# N={n} D={d} M={m} R={r} beam={beam} k={k} queries={nq}");

    print!("[1/6] generating data ... ");
    let t0 = Instant::now();
    let base = gaussian(n, d, 42);
    let queries = gaussian(nq, d, 43);
    println!("{:.2}s", t0.elapsed().as_secs_f64());

    print!("[2/6] training PQ (m={m}, k=256) ... ");
    let t0 = Instant::now();
    let pq = ProductQuantizer::train(&base, d, m, 7);
    println!("{:.2}s", t0.elapsed().as_secs_f64());

    print!("[3/6] encoding {n} codes ... ");
    let t0 = Instant::now();
    let codes = pq.encode_all(&base);
    println!("{:.2}s ({} bytes)", t0.elapsed().as_secs_f64(), codes.len());

    print!("[4/6] building k-NN graph (brute) ... ");
    let t0 = Instant::now();
    let graph = KnnGraph::build_bruteforce(&base, d, r);
    let build_s = t0.elapsed().as_secs_f64();
    println!("{:.2}s (graph RAM={} B)", build_s, graph.ram_bytes());

    print!("[5/6] computing ground truth (brute) ... ");
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = (0..nq)
        .map(|q| brute_force_topk(&base, d, &queries[q * d..(q + 1) * d], k))
        .collect();
    println!("{:.2}s", t0.elapsed().as_secs_f64());

    println!("[6/6] running variants ...");
    let mut rows: Vec<(String, usize, f64, f32)> = Vec::new();

    // Variant 1: flat f32 in RAM
    {
        let backend = FlatF32Ram::new(base.clone(), d);
        rows.push(run_variant("flat-f32-ram", &graph, backend, &queries, d, &truth, k, beam));
    }

    // Variant 2: PQ in RAM
    {
        // Re-train identical PQ so structure is independent (same seed).
        let pq2 = ProductQuantizer::train(&base, d, m, 7);
        let codes2 = pq2.encode_all(&base);
        let backend = PqRam::new(pq2, codes2);
        rows.push(run_variant("pq-ram", &graph, backend, &queries, d, &truth, k, beam));
    }

    // Variant 3: AISAQ (PQ on disk)
    {
        let tmp = std::env::temp_dir().join(format!("ruvector-aisaq-codes-{}.bin", std::process::id()));
        let pq3 = ProductQuantizer::train(&base, d, m, 7);
        let codes3 = pq3.encode_all(&base);
        let backend = PqDisk::create(pq3, &codes3, tmp.clone()).expect("mmap create");
        backend.advise_random().ok();
        rows.push(run_variant("pq-disk-aisaq", &graph, backend, &queries, d, &truth, k, beam));
        let _ = std::fs::remove_file(tmp);
    }

    println!();
    println!("| variant        | RAM (heap) B | per-query us | recall@{k} |");
    println!("|----------------|-------------:|-------------:|-----------:|");
    for (label, ram, us, recall) in &rows {
        println!("| {:<14} | {:>12} | {:>12.2} | {:>10.4} |", label, ram, us, recall);
    }

    // Estimated bytes math (theoretical, exclusive of graph)
    println!();
    println!("Estimated per-point storage:");
    println!("  flat-f32     : {} B  (D*4)", d * 4);
    println!("  pq codes     : {} B  (M)", m);
    println!("  graph edges  : {} B  (R*4)", r * 4);
    println!("Total dataset code-only bytes at N={n}: flat={} MB, pq={} KB",
        (n * d * 4) / (1024 * 1024),
        (n * m) / 1024);
}
