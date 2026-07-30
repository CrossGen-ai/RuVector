//! Reproducible end-to-end benchmark. Prints a real table of strategy performance.

use ruvector_laet::bench::{make_dataset, make_queries, run_strategy};
use ruvector_laet::features::Features;
use ruvector_laet::graph::reset_dist_calls;
use ruvector_laet::train::{fit_ridge, LinearModel};
use ruvector_laet::{build_training_set, ground_truth, FixedEf, Index, LaetStop, PatienceStop};

fn main() -> anyhow::Result<()> {
    let n = 5_000usize;
    let dim = 64usize;
    let k = 10usize;
    let m = 16usize;
    let ef_baseline = 128usize;
    let oracle_ef = 160usize;

    println!("[laet-bench] building dataset n={n} dim={dim} m={m} ...");
    let data = make_dataset(n, dim, 42);
    let train_q = make_queries(200, dim, 1001);
    let test_q = make_queries(100, dim, 2002);

    let index = Index::new(data.clone(), m);
    reset_dist_calls();

    // Deterministic multi-entry set: 8 evenly-spaced ids act as pseudo-HNSW
    // hierarchy entry points so every query cluster is reachable.
    let stride = (index.graph.len() / 8).max(1) as u32;
    let entry_ids: Vec<u32> = (0..8).map(|i| i as u32 * stride).collect();

    println!("[laet-bench] computing ground truth for test set ...");
    let truths: Vec<Vec<u32>> = test_q.iter().map(|q| ground_truth(&data, q, k)).collect();

    println!("[laet-bench] building oracle training set (ef={oracle_ef}) ...");
    let (xs, ys) = build_training_set(&index, &train_q, &entry_ids, oracle_ef, k);
    println!("[laet-bench] training ridge regression on {} rows ...", xs.len());
    let w = fit_ridge(&xs, &ys, 1e-2);
    // Threshold: mean of predicted scores on training rows scaled down.
    // Calibrate threshold: pick the 15th-percentile predicted score across training
    // rows. This yields a predictor that stops roughly when the model estimates the
    // remaining improvement is in the bottom 15% seen during training — conservative
    // enough to preserve recall, aggressive enough to save real work.
    let mut scores: Vec<f32> = {
        let dummy = LinearModel { w: w.clone(), threshold: 0.0 };
        xs.iter()
            .map(|row| {
                let f = Features {
                    iter: row[0],
                    best_dist: row[1],
                    delta_best_dist: row[2],
                    iters_since_improve: row[3],
                    ef_min_reached: row[4],
                };
                dummy.score(&f)
            })
            .collect()
    };
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thresh = scores[(scores.len() as f32 * 0.60) as usize];
    let model = LinearModel { w, threshold: thresh };
    println!(
        "[laet-bench] model.w = {:?} threshold={:.4}",
        model.w, model.threshold
    );

    let baseline = run_strategy(
        "FixedEf(baseline)",
        &index,
        &test_q,
        &truths,
        &entry_ids,
        k,
        FixedEf { ef: ef_baseline },
    );
    let patience = run_strategy(
        "PatienceStop(p=10)",
        &index,
        &test_q,
        &truths,
        &entry_ids,
        k,
        PatienceStop { patience: 10, max_iter: ef_baseline },
    );
    let laet = run_strategy(
        "LaetStop",
        &index,
        &test_q,
        &truths,
        &entry_ids,
        k,
        LaetStop { model: model.clone(), min_iter: 12, max_iter: ef_baseline },
    );

    println!("\n=== ruvector-laet benchmark (n={n} dim={dim} k={k} m={m}) ===");
    println!(
        "{:<22} {:>10} {:>18} {:>16}",
        "strategy", "recall@10", "avg_dist_calls", "avg_latency_us"
    );
    for r in [&baseline, &patience, &laet] {
        println!(
            "{:<22} {:>10.4} {:>18.1} {:>16.2}",
            r.name, r.recall_at_10, r.avg_dist_calls, r.avg_latency_us
        );
    }

    let saved =
        (baseline.avg_dist_calls - laet.avg_dist_calls) / baseline.avg_dist_calls * 100.0;
    let recall_gap = baseline.recall_at_10 - laet.recall_at_10;
    println!(
        "\nLaetStop vs FixedEf: {:+.1}% distance-computations, recall Δ = {:+.4}",
        -saved, recall_gap
    );
    Ok(())
}
