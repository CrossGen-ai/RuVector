//! Benchmark binary — produces the numbers cited in ADR-305 and in the
//! research README. Runs three termination policies on identical
//! (graph, queries, entry-points) inputs so any difference in
//! (recall, visits, latency) is attributable to the policy.
//!
//! Run with:
//!     cargo run --release -p ruvector-topk-stability-terminate --bin topk-stability-bench

use std::time::Instant;

use ruvector_topk_stability_terminate::graph::KnnGraph;
use ruvector_topk_stability_terminate::policy::{
    FixedBudget, GapThreshold, KendallTauStability, TerminationPolicy,
};
use ruvector_topk_stability_terminate::search::search;
use ruvector_topk_stability_terminate::util::{random_unit_vectors, SplitMix64};

struct Row {
    policy: String,
    recall_at_k: f32,
    mean_visits: f32,
    mean_distances: f32,
    early_frac: f32,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run(
    name: &str,
    policy_factory: &dyn Fn() -> Box<dyn TerminationPolicy>,
    graph: &KnnGraph,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    entry_ids: &[u32],
    k: usize,
    ef_max: usize,
) -> Row {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut sum_visits = 0usize;
    let mut sum_dists = 0usize;
    let mut sum_early = 0usize;
    let mut sum_hit = 0usize;
    let mut sum_denom = 0usize;

    let mut policy = policy_factory();
    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let (top, stats) = search(graph, q, entry_ids, k, ef_max, policy.as_mut());
        let us = t.elapsed().as_secs_f64() * 1e6;
        latencies.push(us);
        sum_visits += stats.visits;
        sum_dists += stats.distances;
        if stats.early_stopped { sum_early += 1; }
        let gt = &ground_truth[qi];
        let hit = top.iter().filter(|s| gt.contains(&s.id)).count();
        sum_hit += hit;
        sum_denom += k;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Row {
        policy: name.to_string(),
        recall_at_k: sum_hit as f32 / sum_denom as f32,
        mean_visits: sum_visits as f32 / queries.len() as f32,
        mean_distances: sum_dists as f32 / queries.len() as f32,
        early_frac: sum_early as f32 / queries.len() as f32,
        p50_us: percentile(&latencies, 0.50),
        p95_us: percentile(&latencies, 0.95),
        p99_us: percentile(&latencies, 0.99),
    }
}

fn main() {
    // Config — modest but real, benchmark completes in seconds on Apple M4.
    let n = 10_000usize;
    let dim = 128usize;
    let m = 16usize;
    let n_queries = 500usize;
    let k = 10usize;
    let ef_max = 128usize;
    let seed_data = 20260819u64;
    let seed_query = 424242u64;

    eprintln!(
        "[bench] building index: n={n} dim={dim} m={m}, ef_max={ef_max}, k={k}"
    );
    let t = Instant::now();
    let data = random_unit_vectors(n, dim, seed_data);
    let graph = KnnGraph::build(data, dim, m);
    eprintln!("[bench] index built in {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let queries = random_unit_vectors(n_queries, dim, seed_query);
    let queries: Vec<Vec<f32>> =
        (0..n_queries).map(|i| queries[i * dim..(i + 1) * dim].to_vec()).collect();
    let ground_truth: Vec<Vec<u32>> =
        queries.iter().map(|q| graph.brute_topk(q, k)).collect();
    eprintln!("[bench] ground truth in {:.2}s", t.elapsed().as_secs_f64());

    // Deterministic entry points shared across all policies.
    let mut rng = SplitMix64::new(9);
    let entry_ids: Vec<u32> = (0..4).map(|_| rng.next_range(n as u32)).collect();

    // Warm the graph pages once so the first policy isn't unfairly slow.
    {
        let mut pol: Box<dyn TerminationPolicy> = Box::new(FixedBudget);
        for q in queries.iter().take(50) {
            let _ = search(&graph, q, &entry_ids, k, ef_max, pol.as_mut());
        }
    }

    // Warmup floor: don't allow any policy to terminate before touching
    // enough of the graph to have a plausible top-k. 40 visits ≈ 3× k here,
    // matching what production HNSW implementations use as ef_search floors.
    let min_v = 40usize;
    let rows = vec![
        run("fixed-ef128", &|| Box::new(FixedBudget),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
        run("gap(eps=0.005,w=8,min=40)",
            &|| Box::new(GapThreshold::new(0.005, 8, min_v)),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
        run("gap(eps=0.001,w=12,min=40)",
            &|| Box::new(GapThreshold::new(0.001, 12, min_v)),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
        run("kendall(tau=0.95,w=4,s=3,min=40)",
            &|| Box::new(KendallTauStability::new(0.95, 4, 3, min_v)),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
        run("kendall(tau=0.98,w=4,s=4,min=40)",
            &|| Box::new(KendallTauStability::new(0.98, 4, 4, min_v)),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
        run("kendall(tau=1.00,w=2,s=5,min=40)",
            &|| Box::new(KendallTauStability::new(1.00, 2, 5, min_v)),
            &graph, &queries, &ground_truth, &entry_ids, k, ef_max),
    ];

    println!();
    println!("policy                        recall@10  visits    dists     early%   p50us   p95us   p99us");
    println!("----------------------------  ---------  --------  --------  -------  ------  ------  ------");
    for r in &rows {
        println!(
            "{name:<28}  {rec:>9.4}  {v:>8.1}  {d:>8.1}  {e:>6.1}%  {p50:>6.1}  {p95:>6.1}  {p99:>6.1}",
            name = r.policy,
            rec = r.recall_at_k,
            v = r.mean_visits,
            d = r.mean_distances,
            e = r.early_frac * 100.0,
            p50 = r.p50_us,
            p95 = r.p95_us,
            p99 = r.p99_us,
        );
    }

    // Emit a machine-readable line the research README can quote verbatim.
    println!();
    println!("# n={n} dim={dim} m={m} ef_max={ef_max} k={k} queries={n_queries}");
}
