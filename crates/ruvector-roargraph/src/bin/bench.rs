//! Real benchmark: compares three index variants on synthetic OOD ANN data.
//!
//!   1. `random-graph`  — degree-`M` random graph + beam search (floor).
//!   2. `knn-graph`     — in-base kNN graph + beam search (Vamana init).
//!   3. `roargraph`     — projected bipartite graph built from training queries
//!                        drawn from the *query* distribution.
//!
//! Output is a markdown table; numbers feed the research doc and the gist.
//! Run as: `cargo run --release -p ruvector-roargraph --bin ruvector-roargraph-bench`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_roargraph::{
    brute_knn, KnnGraph, RandomGraph, RoarConfig, RoarGraph, Vector,
};
use std::collections::HashSet;
use std::time::Instant;

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    // Box-Muller. Two uniforms in (0, 1].
    let u1: f32 = (rng.gen::<f32>().max(1e-7)).min(1.0);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn gen_gauss(n: usize, d: usize, mean: f32, seed: u64) -> Vec<Vector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| gaussian(&mut rng) + mean).collect())
        .collect()
}

fn recall_at_k(got: &[(u32, f32)], truth: &[u32], k: usize) -> f32 {
    let truth: HashSet<u32> = truth.iter().take(k).copied().collect();
    let hits = got.iter().take(k).filter(|(id, _)| truth.contains(id)).count();
    hits as f32 / k as f32
}

#[derive(Debug, Clone)]
struct Row {
    label: &'static str,
    build_ms: f64,
    edges: usize,
    avg_deg: f32,
    bytes: usize,
    qps: f64,
    recall_in: f32,
    recall_ood: f32,
}

fn fmt_row(r: &Row) -> String {
    format!(
        "| {:<14} | {:>9.1} | {:>10} | {:>7.2} | {:>10} | {:>10.1} | {:>6.3} | {:>6.3} |",
        r.label, r.build_ms, r.edges, r.avg_deg, r.bytes, r.qps, r.recall_in, r.recall_ood
    )
}

fn main() {
    println!("ruvector-roargraph benchmark");
    println!("Hardware: {}", std::env::consts::ARCH);
    println!();

    // ---- parameters ----
    let n = 4_000;       // base size
    let d = 64;          // embedding dim
    let n_train = 2_000; // training queries (RoarGraph only)
    let n_query_id = 500;  // in-distribution test queries
    let n_query_ood = 500; // out-of-distribution test queries
    let k = 10;
    let ef = 64;
    let degree = 24;
    let nap_k = 24;

    println!(
        "Workload: n={n}, d={d}, train={n_train}, queries(id|ood)={n_query_id}|{n_query_ood}, k={k}, ef={ef}, degree={degree}, nap_k={nap_k}"
    );

    // ---- data ----
    let base = gen_gauss(n, d, 0.0, 11);
    let train_q = gen_gauss(n_train, d, 1.5, 22);     // shifted = query distribution
    let queries_id = gen_gauss(n_query_id, d, 0.0, 33);
    let queries_ood = gen_gauss(n_query_ood, d, 1.5, 44);

    let t = Instant::now();
    let truth_id: Vec<Vec<u32>> = queries_id.iter().map(|q| brute_knn(&base, q, k)).collect();
    let truth_ood: Vec<Vec<u32>> = queries_ood.iter().map(|q| brute_knn(&base, q, k)).collect();
    let truth_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("Brute-force ground truth for {} queries computed in {:.1} ms", n_query_id + n_query_ood, truth_ms);

    let cfg = RoarConfig {
        nap_k,
        degree,
        build_ef: ef,
        alpha: 1.2,
        seed: 7,
    };

    let mut rows: Vec<Row> = Vec::new();

    // 1) random graph baseline
    {
        let t = Instant::now();
        let g = RandomGraph::build(base.clone(), degree, 7);
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let edges = g.edge_count();
        let avg_deg = g.avg_degree();
        let bytes = edges * std::mem::size_of::<u32>() + n * d * std::mem::size_of::<f32>();

        let (qps, r_in, r_ood) = measure(&queries_id, &queries_ood, &truth_id, &truth_ood, k, ef, |q| g.search(q, k, ef));
        rows.push(Row { label: "random-graph", build_ms, edges, avg_deg, bytes, qps, recall_in: r_in, recall_ood: r_ood });
    }

    // 2) in-base kNN graph
    {
        let t = Instant::now();
        let g = KnnGraph::build(base.clone(), cfg.clone());
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let edges = g.edge_count();
        let avg_deg = g.avg_degree();
        let bytes = edges * std::mem::size_of::<u32>() + n * d * std::mem::size_of::<f32>();
        let (qps, r_in, r_ood) = measure(&queries_id, &queries_ood, &truth_id, &truth_ood, k, ef, |q| g.search(q, k, ef));
        rows.push(Row { label: "knn-graph", build_ms, edges, avg_deg, bytes, qps, recall_in: r_in, recall_ood: r_ood });
    }

    // 3) RoarGraph (trained on query-side distribution)
    {
        let t = Instant::now();
        let g = RoarGraph::build(base.clone(), &train_q, cfg.clone());
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let edges = g.edge_count();
        let avg_deg = g.avg_degree();
        let bytes = edges * std::mem::size_of::<u32>() + n * d * std::mem::size_of::<f32>();
        let (qps, r_in, r_ood) = measure(&queries_id, &queries_ood, &truth_id, &truth_ood, k, ef, |q| g.search(q, k, ef));
        rows.push(Row { label: "roargraph", build_ms, edges, avg_deg, bytes, qps, recall_in: r_in, recall_ood: r_ood });
    }

    println!();
    println!("| {:<14} | {:>9} | {:>10} | {:>7} | {:>10} | {:>10} | {:>6} | {:>6} |",
        "variant", "build(ms)", "edges", "avg_deg", "bytes", "qps", "rec_id", "rec_ood");
    println!("|----------------|----------:|-----------:|--------:|-----------:|-----------:|-------:|-------:|");
    for r in &rows {
        println!("{}", fmt_row(r));
    }

    // ---- acceptance criterion ----
    let roar = rows.iter().find(|r| r.label == "roargraph").unwrap();
    let knn = rows.iter().find(|r| r.label == "knn-graph").unwrap();
    let random = rows.iter().find(|r| r.label == "random-graph").unwrap();

    println!();
    println!("Acceptance:");
    let pass_ood = roar.recall_ood >= knn.recall_ood;
    let pass_floor = roar.recall_ood > random.recall_ood;
    let pass_recall = roar.recall_ood >= 0.80;
    println!("  roar.recall_ood ({:.3}) >= knn.recall_ood ({:.3}) : {}", roar.recall_ood, knn.recall_ood, if pass_ood { "PASS" } else { "FAIL" });
    println!("  roar.recall_ood ({:.3}) >  random.recall_ood ({:.3}) : {}", roar.recall_ood, random.recall_ood, if pass_floor { "PASS" } else { "FAIL" });
    println!("  roar.recall_ood ({:.3}) >= 0.80 : {}", roar.recall_ood, if pass_recall { "PASS" } else { "FAIL" });

    if !(pass_ood && pass_floor && pass_recall) {
        std::process::exit(2);
    }
}

fn measure<F>(
    queries_id: &[Vector],
    queries_ood: &[Vector],
    truth_id: &[Vec<u32>],
    truth_ood: &[Vec<u32>],
    k: usize,
    _ef: usize,
    mut search: F,
) -> (f64, f32, f32)
where
    F: FnMut(&[f32]) -> Vec<(u32, f32)>,
{
    let t = Instant::now();
    let mut r_in_sum = 0.0f32;
    for (q, t) in queries_id.iter().zip(truth_id.iter()) {
        let got = search(q);
        r_in_sum += recall_at_k(&got, t, k);
    }
    let mut r_ood_sum = 0.0f32;
    for (q, t) in queries_ood.iter().zip(truth_ood.iter()) {
        let got = search(q);
        r_ood_sum += recall_at_k(&got, t, k);
    }
    let total = queries_id.len() + queries_ood.len();
    let qps = total as f64 / t.elapsed().as_secs_f64();
    let r_in = r_in_sum / queries_id.len() as f32;
    let r_ood = r_ood_sum / queries_ood.len() as f32;
    (qps, r_in, r_ood)
}
