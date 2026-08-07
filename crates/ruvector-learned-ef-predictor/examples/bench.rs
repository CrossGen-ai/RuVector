//! Benchmark: compare fixed-ef, gap-ratio, and learned-linear controllers.
//!
//! Dataset: 40k Gaussian-cluster vectors × 96 dims (deterministic seed).
//! Queries: 800 in-distribution queries. Ground truth via brute force.
//! Metric: recall@10 and QPS. Every number below comes from a real run.

use ruvector_learned_ef_predictor::calibrate::{recall_at_k, EF_LADDER};
use ruvector_learned_ef_predictor::{
    calibrate, make_clusters, make_queries, EfController, FixedEf, GapRatioEf, Hnsw,
};
use std::time::Instant;

fn run_controller(
    idx: &Hnsw,
    ctrl: &dyn EfController,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
) -> (f32, f32, f32, u64) {
    // (mean_recall, qps, mean_ef, total_dists)
    let mut recalls = 0.0f32;
    let mut ef_sum = 0.0f32;
    let mut total_dists: u64 = 0;
    let t0 = Instant::now();
    for (q, truth) in queries.iter().zip(truths.iter()) {
        let probe = idx.probe(q);
        let ef = ctrl.choose_ef(&probe, k);
        ef_sum += ef as f32;
        let (pred, stats) = idx.search(q, k, ef);
        total_dists += stats.dists;
        recalls += recall_at_k(&pred, truth);
    }
    let dt = t0.elapsed().as_secs_f32();
    (
        recalls / queries.len() as f32,
        queries.len() as f32 / dt,
        ef_sum / queries.len() as f32,
        total_dists,
    )
}

fn main() {
    // Config.
    let n_vectors = 40_000;
    let dim = 64;
    let n_clusters = 20;
    let sigma = 1.0f32;
    let k = 10;
    let m = 24;
    let ef_construction = 200;
    let n_queries = 800;
    let n_calib = 200;
    let recall_target = 0.95f32;

    println!("== ruvector-learned-ef-predictor bench ==");
    println!(
        "dataset: {} × {}d, {} clusters, sigma={}",
        n_vectors, dim, n_clusters, sigma
    );

    println!("building index (m={}, ef_construction={})...", m, ef_construction);
    let t0 = Instant::now();
    let vectors = make_clusters(n_vectors, dim, n_clusters, sigma, 42);
    let idx = Hnsw::build(vectors, m, ef_construction, 42);
    println!("  built in {:.2}s", t0.elapsed().as_secs_f32());

    // Query splits.
    let queries = make_queries(n_queries, dim, n_clusters, sigma, 1337);
    let calib_queries = make_queries(n_calib, dim, n_clusters, sigma, 9001);

    println!("computing ground truth for {} queries...", n_queries);
    let t0 = Instant::now();
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| idx.brute_force(q, k)).collect();
    println!("  ground truth in {:.2}s", t0.elapsed().as_secs_f32());

    println!("calibrating learned predictor on {} queries (target recall={})", n_calib, recall_target);
    let learned = calibrate(&idx, &calib_queries, k, recall_target);
    println!("  weights: {:?}", learned.weights());

    // Controllers.
    let fixed_16 = FixedEf { ef: 16 };
    let fixed_32 = FixedEf { ef: 32 };
    let fixed_64 = FixedEf { ef: 64 };
    let fixed_128 = FixedEf { ef: 128 };
    let fixed_256 = FixedEf { ef: 256 };
    let gap = GapRatioEf::default();

    let controllers: Vec<&dyn EfController> = vec![
        &fixed_16, &fixed_32, &fixed_64, &fixed_128, &fixed_256, &gap, &learned,
    ];

    println!();
    println!("EF ladder: {:?}", EF_LADDER);
    println!();
    println!("{:<18} {:>10} {:>10} {:>10} {:>12}", "controller", "mean_ef", "recall@10", "qps", "dists/query");
    println!("{}", "-".repeat(64));
    for c in controllers {
        let (recall, qps, mean_ef, total_dists) = run_controller(&idx, c, &queries, &truths, k);
        println!(
            "{:<18} {:>10.1} {:>10.4} {:>10.1} {:>12.1}",
            c.name(),
            mean_ef,
            recall,
            qps,
            total_dists as f32 / queries.len() as f32,
        );
    }
}
