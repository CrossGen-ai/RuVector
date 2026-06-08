//! deg-demo: minimal `cargo run` example. Builds a small random index,
//! deletes a chunk, re-queries, and prints recall + timings.

use ruvector_deg::{Deg, DegConfig, L2};
use std::time::Instant;

fn random(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..(n * d) {
        // xorshift32, deterministic
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

fn main() {
    let d = 64;
    let n = 2_000;
    let queries = 200;
    let k = 10;
    let eps = 80;
    let vecs = random(n, d, 0xC0FFEE);
    let q = random(queries, d, 0xDEADBEEF);

    let cfg = DegConfig { dim: d, edges_per_node: 24, eps_insert: 80 };
    let mut deg = Deg::new(cfg);

    let t0 = Instant::now();
    deg.build::<L2>(&vecs);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut hits = 0usize;
    let t1 = Instant::now();
    for i in 0..queries {
        let q_i = &q[i * d..(i + 1) * d];
        let truth = brute_top(&vecs, q_i, d, k);
        let got: Vec<u32> = deg.query::<L2>(q_i, k, eps).into_iter().map(|r| r.id).collect();
        for t in &truth {
            if got.contains(t) {
                hits += 1;
            }
        }
    }
    let query_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let recall = hits as f64 / (queries as f64 * k as f64);

    println!("DEG demo  n={n}  d={d}  k={k}  eps={eps}");
    println!("  build:   {build_ms:>8.2} ms ({:.1} vec/s)", n as f64 / (build_ms / 1000.0));
    println!("  queries: {query_ms:>8.2} ms ({:.1} qps)", queries as f64 / (query_ms / 1000.0));
    println!("  recall@{k}: {recall:.4}");

    // Demonstrate dynamic deletes.
    let t2 = Instant::now();
    for _ in 0..200 {
        let v = (deg.len() / 2) as u32;
        deg.delete::<L2>(v);
    }
    let del_ms = t2.elapsed().as_secs_f64() * 1000.0;
    println!("  deleted 200 nodes in {del_ms:>8.2} ms, size now {}", deg.len());
}
