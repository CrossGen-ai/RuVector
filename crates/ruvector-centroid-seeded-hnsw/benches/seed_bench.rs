//! Real bench (no `criterion` dep to keep the crate hermetic).
//!
//! Runs seeder variants over synthetic clustered data and reports
//! recall@1, average greedy hops, average distance-call count, and
//! per-query wall time.

use ruvector_centroid_seeded_hnsw::{
    brute_top1, gen_clusters, CentroidSeeder, KnnGraph, MultiCentroidSeeder, RandomSeeder, Seeder,
};
use std::time::Instant;

fn bench(
    label: &str,
    graph: &KnnGraph,
    seeder: &dyn Seeder,
    queries: &[Vec<f32>],
    truth: &[usize],
) {
    let t0 = Instant::now();
    let mut correct = 0usize;
    let mut hops = 0usize;
    let mut dc = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let entries = seeder.entry_points(q);
        let (best, st) = graph.greedy_search(q, &entries);
        if best == truth[qi] {
            correct += 1;
        }
        hops += st.hops;
        dc += st.distance_calls;
    }
    let elapsed = t0.elapsed();
    let per_q_us = elapsed.as_secs_f64() * 1e6 / queries.len() as f64;
    println!(
        "{:>10} | recall@1={:.3} | avg_hops={:5.2} | avg_dist_calls={:6.2} | {:6.1} us/query | total {:>7.2} ms",
        label,
        correct as f64 / queries.len() as f64,
        hops as f64 / queries.len() as f64,
        dc as f64 / queries.len() as f64,
        per_q_us,
        elapsed.as_secs_f64() * 1e3,
    );
}

fn main() {
    println!("== centroid-seeded HNSW / greedy k-NN graph search ==");
    for (n, d, k_clusters) in [(2_000usize, 32usize, 20usize), (5_000, 64, 32)] {
        println!("\n-- N={} D={} clusters={} --", n, d, k_clusters);
        let data = gen_clusters(n, d, k_clusters, 42);
        let t0 = Instant::now();
        let graph = KnnGraph::build(data.clone(), 16);
        println!("built kNN graph in {:.2} s", t0.elapsed().as_secs_f64());

        let mut random_s = RandomSeeder::new(7);
        let mut cent_s = CentroidSeeder::new(k_clusters, 12, 7);
        let mut mcent_s = MultiCentroidSeeder::new(k_clusters, 4, 12, 7);
        random_s.fit(&data);
        let t_fit = Instant::now();
        cent_s.fit(&data);
        println!("k-means fit ({}c) in {:.2} ms", k_clusters, t_fit.elapsed().as_secs_f64() * 1e3);
        mcent_s.fit(&data);

        let queries = gen_clusters(500, d, k_clusters, 9999);
        let truth: Vec<usize> = queries.iter().map(|q| brute_top1(&data, q)).collect();

        bench("random", &graph, &random_s, &queries, &truth);
        bench("centroid1", &graph, &cent_s, &queries, &truth);
        bench("centroidM", &graph, &mcent_s, &queries, &truth);
    }
}
