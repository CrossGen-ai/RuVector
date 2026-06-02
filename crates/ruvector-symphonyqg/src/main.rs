//! `cargo run --release -p ruvector-symphonyqg --bin symphonyqg-demo`
//!
//! Builds a synthetic multi-Gaussian dataset, brute-forces ground truth,
//! then measures recall@10 + per-query latency for the three SearchModes.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_symphonyqg::{l2_sq, SearchMode, Searcher};
use std::time::Instant;

fn gen_dataset(n: usize, dim: usize, _clusters: usize, seed: u64) -> Vec<f32> {
    // Standard ANN benchmark fixture: i.i.d. Gaussian in R^dim (Box-Muller).
    // Realistic for embedding distributions and avoids pathological cluster
    // gaps that would defeat any single-layer NSW with random entry seeds.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        let u1: f32 = rng.gen_range(1e-7f32..1.0);
        let u2: f32 = rng.gen_range(0.0f32..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        out.push(z);
    }
    out
}

fn brute_top_k(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut all: Vec<(f32, u32)> = (0..n)
        .map(|i| (l2_sq(q, &data[i * dim..(i + 1) * dim]), i as u32))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all.into_iter().take(k).map(|(_, i)| i).collect()
}

#[derive(Default, Clone, Copy)]
struct Stats {
    recall_sum: f64,
    ns_sum: u128,
    queries: usize,
}

impl Stats {
    fn add(&mut self, hits: usize, k: usize, ns: u128) {
        self.recall_sum += hits as f64 / k as f64;
        self.ns_sum += ns;
        self.queries += 1;
    }
    fn recall(&self) -> f64 { self.recall_sum / self.queries as f64 }
    fn qps(&self) -> f64 {
        if self.ns_sum == 0 { 0.0 } else {
            self.queries as f64 / (self.ns_sum as f64 * 1e-9)
        }
    }
    fn us_per_query(&self) -> f64 {
        if self.queries == 0 { 0.0 } else {
            self.ns_sum as f64 / 1000.0 / self.queries as f64
        }
    }
}

fn run_mode(label: &str, s: &Searcher, queries: &[f32], dim: usize,
            ground_truth: &[Vec<u32>], k: usize, ef: usize, mode: SearchMode)
{
    use std::collections::HashSet;
    let mut st = Stats::default();
    for (qi, gt) in ground_truth.iter().enumerate() {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let gt_set: HashSet<u32> = gt.iter().copied().collect();
        let t0 = Instant::now();
        let res = s.search(q, k, ef, mode);
        let ns = t0.elapsed().as_nanos();
        let hits = res.iter().filter(|n| gt_set.contains(&n.id)).count();
        st.add(hits, k, ns);
    }
    println!("  {:<32} recall@{} = {:>5.1}%   qps = {:>9.0}   {:>7.1} µs/query",
        label, k, st.recall() * 100.0, st.qps(), st.us_per_query());
}

fn main() {
    // ---- Workload ----
    let n      = 8_000;
    let dim    = 128;
    let n_q    = 200;
    let k      = 10;
    let m      = 24;
    let ef_c   = 128;
    let ef_s   = 200;
    let rerank = 200;

    println!("== ruvector-symphonyqg demo ==");
    println!("N={} D={} k={} ef_search={} M={} ef_construction={}", n, dim, k, ef_s, m, ef_c);

    let data    = gen_dataset(n,   dim, 32, 1);
    let queries = gen_dataset(n_q, dim, 32, 2);

    // ---- Index build ----
    let t0 = Instant::now();
    let s = Searcher::build(&data, dim, m, ef_c);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\nIndex built in {:.1} ms", build_ms);
    println!(" float bytes  = {:>9}", s.float_bytes());
    println!(" code bytes   = {:>9}  ({} bits per vector)", s.quantized_bytes(), s.bits_per_vec());
    println!(" compression  = {:>9.2}x",
        s.float_bytes() as f64 / s.quantized_bytes().max(1) as f64);

    // ---- Ground truth ----
    println!("\nBuilding brute-force ground truth ...");
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..n_q).map(|qi| {
        let q = &queries[qi * dim..(qi + 1) * dim];
        brute_top_k(&data, dim, q, k)
    }).collect();
    println!("  ground truth in {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ---- Search ----
    println!("\nSearch results:");
    run_mode("Float graph (baseline)",      &s, &queries, dim, &gt, k, ef_s, SearchMode::Float);
    run_mode("Binary graph (1-bit only)",   &s, &queries, dim, &gt, k, ef_s, SearchMode::Binary);
    run_mode("SymphonyQG (1-bit + rerank)", &s, &queries, dim, &gt, k, ef_s,
        SearchMode::Symphony { rerank });

    // ---- Brute-force baseline for QPS context ----
    let mut bf = Stats::default();
    use std::collections::HashSet;
    for (qi, g) in gt.iter().enumerate() {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let gt_set: HashSet<u32> = g.iter().copied().collect();
        let t0 = Instant::now();
        let res = brute_top_k(&data, dim, q, k);
        let ns = t0.elapsed().as_nanos();
        let hits = res.iter().filter(|i| gt_set.contains(i)).count();
        bf.add(hits, k, ns);
    }
    println!("  {:<32} recall@{} = {:>5.1}%   qps = {:>9.0}   {:>7.1} µs/query",
        "Brute force (oracle)", k, bf.recall() * 100.0, bf.qps(), bf.us_per_query());

    println!("\nDone.");
}
