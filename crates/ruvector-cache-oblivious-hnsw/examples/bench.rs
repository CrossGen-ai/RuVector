//! End-to-end benchmark: build the same HNSW under three layouts, then time
//! greedy top-10 search over a shared query set. Reports wall-clock latency,
//! throughput, distance evaluations, and a "slot stride" locality proxy that
//! summarizes memory jumps between successive vector reads.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use ruvector_cache_oblivious_hnsw::{
    build_hnsw, greedy_search, BuildParams, FlatGraph, Layout, SearchParams, DIM,
};

fn gen_queries(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| {
        let mut v = vec![0f32; DIM];
        let mut norm = 0f32;
        for x in v.iter_mut() {
            let mut acc = 0f32;
            for _ in 0..12 { acc += rng.gen::<f32>(); }
            *x = acc - 6.0;
            norm += *x * *x;
        }
        let inv = if norm > 0.0 { 1.0 / norm.sqrt() } else { 1.0 };
        for x in v.iter_mut() { *x *= inv; }
        v
    }).collect()
}

/// Average absolute slot stride between successive visited nodes during
/// search. Smaller = better locality. This is a cheap software proxy for
/// L1 / L2 miss rate that we can measure without perf counters.
fn locality_probe(graph: &FlatGraph, queries: &[Vec<f32>], ef: usize) -> f64 {
    let mut total_stride: u64 = 0;
    let mut total_steps: u64 = 0;
    // Instrumented mini-search that records slot visitation order.
    for q in queries {
        let mut visited = vec![false; graph.n()];
        let entry = graph.entry;
        visited[entry as usize] = true;
        let mut trail: Vec<u32> = vec![entry];
        // frontier
        let ev = graph.vector(entry);
        let d0 = l2_sq(ev, q);
        let mut top: Vec<(f32, u32)> = vec![(d0, entry)];
        let mut cand: Vec<(f32, u32)> = vec![(d0, entry)];
        while let Some((cd, cid)) = pop_min(&mut cand) {
            let worst = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
            if cd > worst && top.len() >= ef { break; }
            for &nb in graph.neighbours_of(cid) {
                if nb == u32::MAX || visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                trail.push(nb);
                let d = l2_sq(graph.vector(nb), q);
                let worst = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
                if top.len() < ef || d < worst {
                    push_sorted(&mut top, (d, nb), ef);
                    cand.push((d, nb));
                }
            }
        }
        for w in trail.windows(2) {
            total_stride += (w[1] as i64 - w[0] as i64).unsigned_abs();
            total_steps += 1;
        }
    }
    if total_steps == 0 { 0.0 } else { total_stride as f64 / total_steps as f64 }
}

#[inline(always)]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..DIM { let d = a[i] - b[i]; s += d * d; }
    s
}

fn pop_min(v: &mut Vec<(f32, u32)>) -> Option<(f32, u32)> {
    if v.is_empty() { return None; }
    let mut best = 0;
    for i in 1..v.len() { if v[i].0 < v[best].0 { best = i; } }
    Some(v.swap_remove(best))
}

fn push_sorted(top: &mut Vec<(f32, u32)>, item: (f32, u32), cap: usize) {
    let pos = top.iter().position(|x| x.0 > item.0).unwrap_or(top.len());
    top.insert(pos, item);
    if top.len() > cap { top.truncate(cap); }
}

fn bench_layout(name: &str, graph: &FlatGraph, queries: &[Vec<f32>], k: usize, ef: usize) {
    // Warmup.
    for q in queries.iter().take(50) {
        let _ = greedy_search(graph, q, SearchParams { k, ef });
    }
    // Timed run.
    let t0 = Instant::now();
    let mut visited_total: u64 = 0;
    let mut dist_total: u64 = 0;
    for q in queries {
        let (_, s) = greedy_search(graph, q, SearchParams { k, ef });
        visited_total += s.visited;
        dist_total += s.distance_evals;
    }
    let elapsed = t0.elapsed();
    let per_q_us = elapsed.as_micros() as f64 / queries.len() as f64;
    let qps = queries.len() as f64 / elapsed.as_secs_f64();
    let stride = locality_probe(graph, queries, ef);
    println!(
        "{:>4} | lat {:>8.2} us/q | qps {:>10.1} | visited/q {:>6.1} | dists/q {:>6.1} | avg-slot-stride {:>10.1}",
        name,
        per_q_us,
        qps,
        visited_total as f64 / queries.len() as f64,
        dist_total as f64 / queries.len() as f64,
        stride,
    );
}

fn main() {
    let params = BuildParams { n: 50_000, m: 16, ef_construction: 96, seed: 0xC0FFEE };
    let queries = gen_queries(1_000, 0xBEEF);
    let k = 10;
    let ef = 64;

    println!("Cache-oblivious HNSW layout benchmark");
    println!(
        "n={} dim={} M={} ef_construction={} queries={} k={} ef={}",
        params.n, DIM, params.m, params.ef_construction, queries.len(), k, ef
    );
    println!("---");

    for layout in [Layout::Bfs, Layout::Dfs, Layout::Veb] {
        let t_build = Instant::now();
        let g = build_hnsw(params, layout);
        let build_s = t_build.elapsed().as_secs_f64();
        println!("built {:?} in {:.2}s (vectors={} bytes, neighbours={} bytes)",
            layout,
            build_s,
            g.vectors.len() * 4,
            g.neighbours.len() * 4,
        );
        let name = match layout { Layout::Bfs => "BFS", Layout::Dfs => "DFS", Layout::Veb => "vEB" };
        bench_layout(name, &g, &queries, k, ef);
    }
}
