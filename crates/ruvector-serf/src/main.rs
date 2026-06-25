//! Benchmark binary. Run with `cargo run --release -p ruvector-serf --bin serf-bench`.
//! Prints a Markdown-friendly table of build time, mean QPS, and mean recall@10
//! for each backend across multiple range-width settings.

use std::time::Instant;

use ruvector_serf::data::Dataset;
use ruvector_serf::graph::Graph;
use ruvector_serf::linear::LinearPrefilter;
use ruvector_serf::post_filter::PostFilterNsw;
use ruvector_serf::serf::SerfIndex;
use ruvector_serf::{recall_at_k, RangeAnn};

const N: usize = 10_000;
const DIM: usize = 64;
const K: usize = 10;
const KGRAPH: usize = 16;
const EF: usize = 64;
const NUM_QUERIES: usize = 200;
const RANGE_FRACS: &[f32] = &[0.01, 0.05, 0.20, 1.00];

fn main() {
    println!("# ruvector-serf benchmark");
    println!(
        "N={N} dim={DIM} kgraph={KGRAPH} ef={EF} k={K} queries={NUM_QUERIES}"
    );

    let t0 = Instant::now();
    let ds = Dataset::random_gaussian(N, DIM, 42);
    let t_data = t0.elapsed();
    println!("dataset built in {:.2?}", t_data);

    let t0 = Instant::now();
    let graph = Graph::build_knn(&ds.points, KGRAPH);
    let t_graph = t0.elapsed();
    let bytes = graph.estimated_bytes();
    let edges = graph.num_edges();
    println!(
        "k-NN graph built in {:.2?} | {edges} undirected edges | {:.2} MiB",
        t_graph,
        bytes as f64 / (1024.0 * 1024.0)
    );

    let truth_idx = LinearPrefilter::new(&ds);
    let post = PostFilterNsw::new(&ds, graph.clone(), EF);
    let serf = SerfIndex::new(&ds, graph, EF);

    println!("\n| range_frac | backend            | mean_qps    | recall@10 |");
    println!("|-----------:|--------------------|------------:|---------:|");

    for &rf in RANGE_FRACS {
        let queries = ds.random_queries(NUM_QUERIES, rf, 99);
        // Ground truth (used both as a baseline timing and as recall reference).
        let (lin_qps, lin_results) = run(&truth_idx, &queries);
        // Recall of linear-prefilter against itself is 1.0 by construction.
        report(rf, "linear-prefilter", lin_qps, 1.0);

        let (post_qps, post_results) = run(&post, &queries);
        let post_recall = mean_recall(&lin_results, &post_results);
        report(rf, "post-filter-nsw", post_qps, post_recall);

        let (serf_qps, serf_results) = run(&serf, &queries);
        let serf_recall = mean_recall(&lin_results, &serf_results);
        report(rf, "serf-edge-pruned", serf_qps, serf_recall);
    }

    println!("\nDone.");
}

fn run<A: RangeAnn>(
    idx: &A,
    queries: &[ruvector_serf::data::Query],
) -> (f64, Vec<Vec<ruvector_serf::Hit>>) {
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        out.push(idx.search(q, K));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    (qps, out)
}

fn mean_recall(
    truth: &[Vec<ruvector_serf::Hit>],
    cand: &[Vec<ruvector_serf::Hit>],
) -> f32 {
    let sum: f32 = truth
        .iter()
        .zip(cand.iter())
        .map(|(t, c)| recall_at_k(t, c, K))
        .sum();
    sum / truth.len() as f32
}

fn report(rf: f32, name: &str, qps: f64, recall: f32) {
    println!(
        "| {:>9.2}% | {:<18} | {:>11.1} | {:>8.3} |",
        rf * 100.0,
        name,
        qps,
        recall
    );
}
