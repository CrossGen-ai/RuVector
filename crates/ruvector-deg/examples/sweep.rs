//! Sweep three (edges_per_node, eps_insert) variants and print a comparison
//! table the research doc consumes verbatim.

use ruvector_deg::{Deg, DegConfig, L2};
use std::time::Instant;

fn random(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..(n * d) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s & 0xFFFFFF) as f32 / 16_777_216.0;
        out.push(u * 2.0 - 1.0);
    }
    out
}

fn brute_top(vecs: &[f32], q: &[f32], d: usize, k: usize) -> Vec<u32> {
    let n = vecs.len() / d;
    let mut all: Vec<(f32, u32)> = (0..n)
        .map(|i| {
            let s: f32 = (0..d)
                .map(|j| {
                    let dd = vecs[i * d + j] - q[j];
                    dd * dd
                })
                .sum();
            (s, i as u32)
        })
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all.into_iter().take(k).map(|(_, id)| id).collect()
}

fn run(name: &str, cfg: DegConfig, vecs: &[f32], q: &[f32], n: usize, queries: usize, k: usize) {
    let d = cfg.dim;
    let eps_q = (cfg.eps_insert / 2).max(k * 4);

    let mut deg = Deg::new(cfg);
    let t0 = Instant::now();
    deg.build::<L2>(vecs);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut hits = 0usize;
    let t1 = Instant::now();
    for i in 0..queries {
        let qi = &q[i * d..(i + 1) * d];
        let truth = brute_top(vecs, qi, d, k);
        let got: Vec<u32> = deg
            .query::<L2>(qi, k, eps_q)
            .into_iter()
            .map(|r| r.id)
            .collect();
        for t in &truth {
            if got.contains(t) {
                hits += 1;
            }
        }
    }
    let query_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let recall = hits as f64 / (queries as f64 * k as f64);

    let mem_est = (n * d * 4) + (n * cfg.edges_per_node * 4);
    let mem_mb = mem_est as f64 / (1024.0 * 1024.0);

    let t2 = Instant::now();
    let to_del = (deg.len() / 4).min(500);
    for _ in 0..to_del {
        let v = (deg.len() / 2) as u32;
        deg.delete::<L2>(v);
    }
    let del_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let del_per = del_ms / to_del as f64;

    println!(
        "{:<9} | M={:>2} eps_i={:>3} | build {:>7.1} ms ({:>6.0} v/s) | query {:>6.1} ms ({:>6.0} qps) | recall@{} {:.4} | mem {:>5.2} MB | delete {:>5.2} ms/op",
        name,
        cfg.edges_per_node,
        cfg.eps_insert,
        build_ms,
        n as f64 / (build_ms / 1000.0),
        query_ms,
        queries as f64 / (query_ms / 1000.0),
        k,
        recall,
        mem_mb,
        del_per
    );
}

fn main() {
    let d = 64;
    let n = 2_000;
    let queries = 200;
    let k = 10;
    let vecs = random(n, d, 0xC0FFEE);
    let q = random(queries, d, 0xDEADBEEF);

    println!("DEG sweep  n={}  d={}  queries={}  k={}", n, d, queries, k);
    println!(
        "{:-<9}-+-{:-<14}-+-{:-<32}-+-{:-<28}-+-{:-<16}-+-{:-<10}-+-{:-<18}",
        "", "", "", "", "", "", ""
    );
    run("baseline", DegConfig { dim: d, edges_per_node: 16, eps_insert: 40 }, &vecs, &q, n, queries, k);
    run("balanced", DegConfig { dim: d, edges_per_node: 24, eps_insert: 80 }, &vecs, &q, n, queries, k);
    run("recall  ", DegConfig { dim: d, edges_per_node: 32, eps_insert: 160 }, &vecs, &q, n, queries, k);
}
