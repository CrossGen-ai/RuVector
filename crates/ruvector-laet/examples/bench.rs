//! `cargo run --release -p ruvector-laet --example bench`
//!
//! Runs all three search strategies on a synthetic dataset and prints
//! recall, distance-call count, and wall-clock latency. The numbers
//! quoted in docs/research/nightly/2026-06-20-laet-hnsw/README.md and
//! in ADR-264 come from running this binary on the developer's
//! laptop; reproduce locally to see your own hardware's results.

use ruvector_laet::data::brute_topk;
use ruvector_laet::search::{calibrate_training_set, recall_at_k};
use ruvector_laet::{
    gen_clustered, FixedEfStrategy, GapHeuristicStrategy, Hnsw, HnswParams, LaetStrategy,
    RidgePredictor, SearchStrategy,
};
use std::time::Instant;

fn bench_strategy<S: SearchStrategy>(
    s: &S,
    idx: &Hnsw,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
) -> (f32, f64, f64, f64) {
    let mut total_recall = 0.0;
    let mut total_dists = 0u64;
    let mut total_ef = 0u64;
    let t = Instant::now();
    for (i, q) in queries.iter().enumerate() {
        let r = s.search(idx, q, k);
        total_recall += recall_at_k(&r.ids, &gt[i], k);
        total_dists += r.dist_calls as u64;
        total_ef += r.ef_used as u64;
    }
    let elapsed_us = t.elapsed().as_secs_f64() * 1e6;
    let nq = queries.len() as f64;
    (
        total_recall / queries.len() as f32,
        total_dists as f64 / nq,
        elapsed_us / nq,
        total_ef as f64 / nq,
    )
}

fn main() {
    let n_base = 50_000usize;
    let n_query = 1_000usize;
    let n_calib = 500usize;
    let dim = 64usize;
    let k = 10usize;

    println!("[laet-hnsw] generating dataset n={n_base} d={dim} k={k} …");
    let ds = gen_clustered(0xBEEF, n_base, n_query + n_calib, dim, 32, 8.0, 1.0);

    println!("[laet-hnsw] building HNSW M=16 efC=100 …");
    let mut idx = Hnsw::new(
        dim,
        HnswParams {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ml: 1.0 / (16f32).ln(),
            seed: 0xC0FFEE,
        },
    );
    let t_build = Instant::now();
    for v in ds.base.iter().cloned() {
        idx.insert(v);
    }
    let build_s = t_build.elapsed().as_secs_f64();
    println!("[laet-hnsw] build done: {:.2}s", build_s);

    let calib_qs = &ds.queries[..n_calib];
    let test_qs = &ds.queries[n_calib..];

    println!("[laet-hnsw] computing ground truth (brute) …");
    let gt_all: Vec<Vec<u32>> = ds
        .queries
        .iter()
        .map(|q| {
            brute_topk(&ds.base, q, k)
                .into_iter()
                .map(|(i, _)| i as u32)
                .collect()
        })
        .collect();
    let gt_calib = &gt_all[..n_calib];
    let gt_test = &gt_all[n_calib..];

    println!("[laet-hnsw] calibrating LAET predictor on {} queries …", n_calib);
    let target_recall = 0.95f32;
    let ef_grid = [8usize, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    let train = calibrate_training_set(&idx, calib_qs, gt_calib, k, target_recall, &ef_grid);
    let xs: Vec<[f64; 5]> = train.iter().map(|(f, _)| f.to_vec()).collect();
    let ys: Vec<f64> = train.iter().map(|(_, y)| *y).collect();

    let mean_target: f64 = ys.iter().sum::<f64>() / ys.len() as f64;
    let var_target: f64 = ys.iter().map(|y| (y - mean_target).powi(2)).sum::<f64>() / ys.len() as f64;
    println!(
        "[laet-hnsw] calibrated targets: mean={:.1}  std={:.1}",
        mean_target,
        var_target.sqrt()
    );

    let mut pred = RidgePredictor {
        lambda: 1.0,
        ef_floor: k,
        ef_ceil: 256,
        ..Default::default()
    };
    pred.fit(&xs, &ys);
    println!("[laet-hnsw] ridge weights: {:?}", pred.weights);

    println!();
    println!("=== Benchmark ({} queries, k={}) ===", test_qs.len(), k);
    println!(
        "{:<20} {:>10} {:>14} {:>14} {:>10}",
        "strategy", "recall", "dist/query", "us/query", "ef/query"
    );

    let baselines = [16usize, 32, 64, 128, 256];
    for ef in baselines {
        let s = FixedEfStrategy { ef };
        let (r, d, lat, ef_used) = bench_strategy(&s, &idx, test_qs, gt_test, k);
        println!(
            "fixed-ef ef={:<8} {:>10.4} {:>14.1} {:>14.2} {:>10.1}",
            ef, r, d, lat, ef_used
        );
    }

    let gh = GapHeuristicStrategy {
        ef_max: 256,
        patience: 4,
        eps: 1e-3,
        ef_floor: k,
    };
    let (r, d, lat, ef_used) = bench_strategy(&gh, &idx, test_qs, gt_test, k);
    println!(
        "gap-heuristic         {:>10.4} {:>14.1} {:>14.2} {:>10.1}",
        r, d, lat, ef_used
    );

    let laet = LaetStrategy { predictor: pred };
    let (r, d, lat, ef_used) = bench_strategy(&laet, &idx, test_qs, gt_test, k);
    println!(
        "laet                  {:>10.4} {:>14.1} {:>14.2} {:>10.1}",
        r, d, lat, ef_used
    );

    println!();
    println!("Target recall for LAET calibration: {:.2}", target_recall);
}
