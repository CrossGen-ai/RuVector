//! Real-numbers benchmark binary (not criterion — we want raw deterministic stats).
//!
//! Run: `cargo run -p ruvector-tribase --bin tribase_bench --release`.

use std::time::Instant;
use ruvector_tribase::*;

struct Row {
    name: &'static str,
    n: usize,
    d: usize,
    ef: usize,
    k: usize,
    build_ms: f64,
    qps: f64,
    full_dist_per_q: f64,
    pruned_per_q: f64,
    recall: f64,
}

fn run_one<S: AnnSearcher>(name: &'static str, searcher: &S, graph: &FlatGraph,
                           queries: &[Vec<f32>], truth: &[Vec<u32>], k: usize, ef: usize,
                           n: usize, d: usize, build_ms: f64) -> Row {
    let mut total = SearchStats::default();
    let mut recall = 0.0f64;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let mut s = SearchStats::default();
        let r = searcher.search(q, k, ef, &mut s);
        total.merge(&s);
        let truth_set: std::collections::HashSet<u32> = truth[qi].iter().copied().collect();
        let hits = r.iter().filter(|(i, _)| truth_set.contains(i)).count();
        recall += hits as f64 / k as f64;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qcount = queries.len();
    Row {
        name, n, d, ef, k,
        build_ms,
        qps: qcount as f64 / elapsed,
        full_dist_per_q: total.full_dist as f64 / qcount as f64,
        pruned_per_q:    total.pruned as f64 / qcount as f64,
        recall: recall / qcount as f64,
    }
}

fn bench_at(n: usize, d: usize, m: usize, q: usize, k: usize, ef: usize, seed: u64) -> Vec<Row> {
    use std::time::Instant;
    let t0 = Instant::now();
    let g = FlatGraph::random_knn(n, d, m, seed);
    let build_graph_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Synthetic query set: held-out points generated from the same generator.
    let g_query = FlatGraph::random_knn(q.min(n), d, 1, seed.wrapping_add(7));
    let queries: Vec<Vec<f32>> = (0..q as u32)
        .map(|i| g_query.vector(i % g_query.len() as u32).to_vec())
        .collect();
    let truth: Vec<Vec<u32>> = queries.iter()
        .map(|q| brute_force(&g, q, k).into_iter().map(|(i, _)| i).collect())
        .collect();

    let base = BaselineSearcher { graph: &g, entry: 0 };
    let t1 = Instant::now();
    let tri1 = TribaseSearcher::build(&g, 0);
    let build_tri1 = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let trik = MultiLandmarkSearcher::build(&g, 0, 8, 99);
    let build_trik = t2.elapsed().as_secs_f64() * 1000.0;

    vec![
        run_one("baseline",  &base, &g, &queries, &truth, k, ef, n, d, build_graph_ms),
        run_one("tribase-1", &tri1, &g, &queries, &truth, k, ef, n, d, build_graph_ms + build_tri1),
        run_one("tribase-8", &trik, &g, &queries, &truth, k, ef, n, d, build_graph_ms + build_trik),
    ]
}

fn main() {
    println!("{:>10} | {:>5} | {:>4} | {:>4} | {:>3} | {:>9} | {:>9} | {:>12} | {:>10} | {:>6}",
        "name", "n", "d", "ef", "k", "build_ms", "qps", "fullD/q", "pruned/q", "recall");
    println!("{}", "-".repeat(110));

    let configs = [
        (1_000, 32, 16, 100, 10, 32),
        (1_000, 32, 16, 100, 10, 64),
        (2_000, 64, 16, 100, 10, 64),
        (4_000, 128, 16, 100, 10, 64),
    ];

    for (n, d, m, q, k, ef) in configs {
        let rows = bench_at(n, d, m, q, k, ef, 7);
        for r in rows {
            println!("{:>10} | {:>5} | {:>4} | {:>4} | {:>3} | {:>9.1} | {:>9.1} | {:>12.1} | {:>10.1} | {:>6.3}",
                r.name, r.n, r.d, r.ef, r.k, r.build_ms, r.qps, r.full_dist_per_q, r.pruned_per_q, r.recall);
        }
        println!();
    }
}
