//! Benchmark: FixedEf vs LearnedTermination vs Oracle across clustered data.
//!
//! Runs a train/test split, learns the logistic predictor on the train split,
//! then measures recall + distance-call cost + wall-clock time on the test split.
//! All numbers are real cargo-run measurements.

use ruvector_learned_termination::{
    dataset::{clustered_vectors, ground_truth},
    graph::{FlatGraph, GraphConfig},
    predictor::{LogisticPredictor, TrainConfig},
    recall_at_k,
    search::{
        collect_training_samples, FixedEfSearch, LearnedTermination, OracleTermination, Searcher,
    },
};
use std::time::Instant;

fn median(xs: &mut [usize]) -> usize {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn main() {
    // Config chosen so total runtime is under 30s on a laptop.
    const N: usize = 2000;
    const DIM: usize = 32;
    const CLUSTERS: usize = 10;
    const NOISE: f32 = 0.2;
    const K_GRAPH: usize = 20;
    const K: usize = 10;
    const EF: usize = 80;
    const TRAIN_QUERIES: usize = 60;
    const TEST_QUERIES: usize = 200;

    println!("=== ruvector-learned-termination benchmark ===");
    println!(
        "corpus={N} dim={DIM} clusters={CLUSTERS} noise={NOISE} \
         k_graph={K_GRAPH} k={K} ef={EF} \
         train_q={TRAIN_QUERIES} test_q={TEST_QUERIES}"
    );

    let t_build = Instant::now();
    let corpus = clustered_vectors(N, DIM, CLUSTERS, NOISE, 42);
    let graph = FlatGraph::build(
        corpus.clone(),
        GraphConfig {
            k_neighbours: K_GRAPH,
        },
    );
    println!("built graph in {:.2?}", t_build.elapsed());

    // Train/test split by seed offset.
    let train_queries: Vec<Vec<f32>> = clustered_vectors(TRAIN_QUERIES, DIM, CLUSTERS, NOISE, 7);
    let test_queries: Vec<Vec<f32>> = clustered_vectors(TEST_QUERIES, DIM, CLUSTERS, NOISE, 8);

    // Collect training samples across all train queries.
    let t_collect = Instant::now();
    let mut samples = Vec::new();
    for q in &train_queries {
        samples.extend(collect_training_samples(&graph, q, EF, K));
    }
    let pos = samples.iter().filter(|(_, y)| *y > 0.5).count();
    let neg = samples.iter().filter(|(_, y)| *y <= 0.5).count();
    println!(
        "collected {} training samples ({} pos / {} neg) in {:.2?}",
        samples.len(),
        pos,
        neg,
        t_collect.elapsed()
    );

    let t_train = Instant::now();
    let predictor = LogisticPredictor::train(&samples, &TrainConfig::default());
    println!(
        "trained predictor in {:.2?}: weights = {:?}",
        t_train.elapsed(),
        predictor.w
    );

    // Build searchers.
    let fixed = FixedEfSearch {
        graph: &graph,
        ef_search: EF,
    };
    let learned_015 = LearnedTermination {
        graph: &graph,
        ef_search: EF,
        predictor: predictor.clone(),
        tau: 0.15,
        improve_window: 4,
        min_steps: 6,
    };
    let learned_030 = LearnedTermination {
        graph: &graph,
        ef_search: EF,
        predictor: predictor.clone(),
        tau: 0.30,
        improve_window: 4,
        min_steps: 6,
    };
    let oracle = OracleTermination {
        graph: &graph,
        ef_search: EF,
        patience: 3,
        target: std::sync::Mutex::new(None),
    };

    // Evaluate each searcher.
    let variants: [(&str, &dyn Fn(&[f32]) -> (Vec<_>, _)); 4] = [
        ("Fixed(ef=80)", &|q| fixed.search(q, K)),
        ("Learned(tau=0.15)", &|q| learned_015.search(q, K)),
        ("Learned(tau=0.30)", &|q| learned_030.search(q, K)),
        ("Oracle(patience=3)", &|q| {
            let gt = ground_truth(q, &corpus, K);
            *oracle.target.lock().unwrap() = Some(gt);
            oracle.search(q, K)
        }),
    ];

    println!(
        "\n{:<22} | {:>10} | {:>16} | {:>13} | {:>12} | {:>10}",
        "variant", "recall@10", "mean_beam_dist", "median_steps", "mean_us", "speedup"
    );
    println!("(beam dist calls only — entry-scan cost is constant across variants)");
    println!("{}", "-".repeat(96));

    let mut fixed_dist = 1usize;

    for (name, run) in &variants {
        let mut total_recall = 0.0f32;
        let mut total_dist = 0usize;
        let mut steps_v: Vec<usize> = Vec::with_capacity(test_queries.len());
        let t = Instant::now();
        for q in &test_queries {
            let gt = ground_truth(q, &corpus, K);
            let (hits, stats) = run(q);
            total_recall += recall_at_k(&gt, &hits, K);
            total_dist += stats.dist_calls;
            steps_v.push(stats.steps);
        }
        let elapsed = t.elapsed();
        let n = test_queries.len() as f32;
        let recall = total_recall / n;
        let mean_dist = (total_dist as f32) / n;
        let median_steps = median(&mut steps_v);
        let mean_us = elapsed.as_micros() as f32 / n;
        if *name == "Fixed(ef=80)" {
            fixed_dist = total_dist.max(1);
        }
        let speedup = fixed_dist as f32 / total_dist.max(1) as f32;
        println!(
            "{:<22} | {:>10.4} | {:>16.1} | {:>13} | {:>12.1} | {:>9.2}x",
            name, recall, mean_dist, median_steps, mean_us, speedup
        );
    }

    println!("\nnote: 'speedup' is (Fixed dist-calls) / (variant dist-calls). \
              Oracle is a lower bound on dist calls, not deployable in production.");
}
