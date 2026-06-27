//! End-to-end benchmark for the learned `ef_search` budget predictor.
//!
//! Workload: a deliberately heterogeneous synthetic distribution — 60% of
//! queries are drawn from the same Gaussian mixture as the corpus (easy
//! queries that any tiny `ef` will solve) and 40% are drawn from a
//! shifted/spread-out mixture (out-of-distribution queries that need much
//! more search effort). This mirrors the OoD workloads called out by
//! RoarGraph (VLDB 2024) and the per-query difficulty story from
//! "Steiner-hardness" (NeurIPS 2024).
//!
//! Reported variants:
//!   - baseline-fixed:      one static `ef` calibrated to hit the avg recall
//!   - baseline-pessimist:  one static `ef` calibrated to hit per-query target
//!   - oracle:              per-query smallest `ef` from the binary-search ladder
//!   - learned (this work): ridge-regression predictor with safety margin
//!
//! All numbers below come from `cargo run --release -p ruvector-learned-ef-budget`.

use ruvector_learned_ef_budget::features::{extract_features, Medoids, QueryFeatures};
use ruvector_learned_ef_budget::hnsw::{brute_knn, Hnsw, HnswParams, SearchStats};
use ruvector_learned_ef_budget::predictor::{BudgetPredictor, OracleBudget, PredictorConfig};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

const DIM: usize = 64;
const N_CORPUS: usize = 20_000;
const N_TRAIN: usize = 1_500;
const N_TEST: usize = 1_000;
const K: usize = 10;
const TARGET_RECALL: f32 = 0.95;
const N_MEDOIDS: usize = 16;

fn make_mixture(n: usize, dim: usize, n_clusters: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
        .collect();
    let mut out = vec![0.0_f32; n * dim];
    for i in 0..n {
        let c = &centers[i % n_clusters];
        for d in 0..dim {
            out[i * dim + d] = c[d] + rng.gen_range(-spread..spread);
        }
    }
    out
}

fn make_queries(n_in: usize, n_ood: usize, dim: usize, corpus: &[f32], seed: u64) -> Vec<f32> {
    // Heterogeneous jitter regime: every query is anchored to a corpus point
    // (so its true neighbours exist and are reachable by HNSW), but the
    // jitter magnitude varies — easy queries land near a tight cluster and
    // hard ones drift toward cluster boundaries where the local density is
    // low. This is the "Steiner-hardness" workload: hardness is per-query
    // and continuous, not a hard in/out-of-distribution split.
    let mut rng = StdRng::seed_from_u64(seed);
    let total = n_in + n_ood;
    let mut out = vec![0.0_f32; total * dim];
    let n_corp = corpus.len() / dim;
    // Three difficulty tiers — easy/medium/hard ≈ 50/30/20 split
    let easy_end = total / 2;
    let med_end = easy_end + total * 3 / 10;
    for i in 0..total {
        let scale = if i < easy_end {
            rng.gen_range(0.005_f32..0.04)
        } else if i < med_end {
            rng.gen_range(0.05_f32..0.10)
        } else {
            rng.gen_range(0.15_f32..0.30)
        };
        let src = rng.gen_range(0..n_corp);
        let base = i * dim;
        for d in 0..dim {
            out[base + d] = corpus[src * dim + d] + rng.gen_range(-scale..scale);
        }
    }
    // shuffle so test ordering is randomised
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);
    let mut shuffled = vec![0.0_f32; total * dim];
    for (new_i, &old_i) in indices.iter().enumerate() {
        shuffled[new_i * dim..(new_i + 1) * dim]
            .copy_from_slice(&out[old_i * dim..(old_i + 1) * dim]);
    }
    shuffled
}

fn slice<'a>(buf: &'a [f32], i: usize, dim: usize) -> &'a [f32] {
    &buf[i * dim..(i + 1) * dim]
}

#[derive(Default, Clone, Copy, Debug)]
struct Agg {
    queries: u64,
    recall_sum: f64,
    dist_sum: u64,
    ef_sum: u64,
    feat_dists: u64,
    target_hits: u64,
    elapsed_us: u128,
}
impl Agg {
    fn rec(&mut self, recall: f32, dists: u64, ef: usize, feat_d: u64, micros: u128) {
        self.queries += 1;
        self.recall_sum += recall as f64;
        self.dist_sum += dists;
        self.ef_sum += ef as u64;
        self.feat_dists += feat_d;
        self.elapsed_us += micros;
        if recall + 1e-6 >= TARGET_RECALL {
            self.target_hits += 1;
        }
    }
    fn print(&self, name: &str) {
        let q = self.queries.max(1) as f64;
        println!(
            "  {:<22} mean_recall={:.4}  target_hit_rate={:.3}  mean_dists={:>7.1}  mean_ef={:>6.2}  feat_dists/query={:>5.2}  mean_us={:>5.1}",
            name,
            self.recall_sum / q,
            self.target_hits as f64 / q,
            self.dist_sum as f64 / q,
            self.ef_sum as f64 / q,
            self.feat_dists as f64 / q,
            self.elapsed_us as f64 / q,
        );
    }
    fn json(&self, name: &str) -> serde_json::Value {
        let q = self.queries.max(1) as f64;
        serde_json::json!({
            "variant": name,
            "queries": self.queries,
            "mean_recall": self.recall_sum / q,
            "target_hit_rate": self.target_hits as f64 / q,
            "mean_dists": self.dist_sum as f64 / q,
            "mean_ef": self.ef_sum as f64 / q,
            "feat_dists_per_query": self.feat_dists as f64 / q,
            "mean_microseconds": self.elapsed_us as f64 / q,
        })
    }
}

fn recall_for(res: &[(u32, f32)], gt: &HashSet<u32>) -> f32 {
    let hit = res.iter().filter(|(i, _)| gt.contains(i)).count() as f32;
    hit / (res.len().max(1) as f32)
}

fn main() {
    let t_total = Instant::now();
    println!("=== ruvector-learned-ef-budget benchmark ===");
    println!(
        "dim={} corpus={} train_q={} test_q={} k={} target_recall={}",
        DIM, N_CORPUS, N_TRAIN, N_TEST, K, TARGET_RECALL
    );

    // 1. corpus + queries
    let t = Instant::now();
    let corpus = make_mixture(N_CORPUS, DIM, 32, 0.15, 0xC0FFEE);
    let train_q = make_queries(N_TRAIN * 6 / 10, N_TRAIN * 4 / 10, DIM, &corpus, 0xA11CE);
    let test_q = make_queries(N_TEST * 6 / 10, N_TEST * 4 / 10, DIM, &corpus, 0xB0B);
    println!("data ready in {} ms", t.elapsed().as_millis());

    // 2. build HNSW
    let t = Instant::now();
    let mut idx = Hnsw::new(DIM, HnswParams::default());
    for i in 0..N_CORPUS {
        idx.insert(slice(&corpus, i, DIM));
    }
    println!("hnsw built in {} ms (M={} efC={})",
        t.elapsed().as_millis(), idx.params.m, idx.params.ef_construction);

    // 3. medoids for features
    let t = Instant::now();
    let medoids = Medoids::fit(&corpus, DIM, N_MEDOIDS, 0xFEED);
    println!("medoids fit in {} ms ({} centres)", t.elapsed().as_millis(), medoids.k());

    // 4. oracle ladder for labels
    let ladder: Vec<usize> = vec![8, 16, 32, 64, 128, 256, 512];
    let cfg = PredictorConfig {
        target_recall: TARGET_RECALL,
        k: K,
        ef_min: 8,
        ef_max: 512,
        ridge_lambda: 1e-2,
    };

    // 5. build training set (oracle labels + features)
    let t = Instant::now();
    let mut train_feats: Vec<QueryFeatures> = Vec::with_capacity(N_TRAIN);
    let mut train_labels: Vec<usize> = Vec::with_capacity(N_TRAIN);
    for i in 0..N_TRAIN {
        let q = slice(&train_q, i, DIM);
        let mut s = SearchStats::default();
        let f = extract_features(q, &medoids, &idx, &mut s);
        let ef = OracleBudget::label(&idx, &corpus, DIM, q, &cfg, &ladder);
        train_feats.push(f);
        train_labels.push(ef);
    }
    let mut hist = std::collections::BTreeMap::new();
    for &l in &train_labels {
        *hist.entry(l).or_insert(0usize) += 1;
    }
    println!(
        "train labels built in {} ms — oracle ef histogram: {:?}",
        t.elapsed().as_millis(), hist
    );

    // 6. fit predictor with a small log2 safety margin
    let t = Instant::now();
    let predictor = BudgetPredictor::fit(&train_feats, &train_labels, cfg.clone(), 0.6)
        .expect("fit must succeed");
    println!("predictor fit in {} ms, weights = {:?}", t.elapsed().as_millis(), predictor.w);

    // 7. choose baselines from training oracle distribution
    let mut sorted = train_labels.clone();
    sorted.sort_unstable();
    // baseline-fixed: median oracle ef (will miss target on hard queries)
    let baseline_fixed_ef = sorted[sorted.len() / 2];
    // baseline-pessimist: 95th-percentile oracle ef
    let baseline_pessimist_ef = sorted[(sorted.len() as f64 * 0.95) as usize];
    println!(
        "baseline ef chosen — fixed={} pessimist={}",
        baseline_fixed_ef, baseline_pessimist_ef
    );

    // 8. evaluate on test set
    let mut a_fixed = Agg::default();
    let mut a_pess = Agg::default();
    let mut a_oracle = Agg::default();
    let mut a_learned = Agg::default();

    for i in 0..N_TEST {
        let q = slice(&test_q, i, DIM);
        let gt: HashSet<u32> =
            brute_knn(&corpus, DIM, q, K).into_iter().map(|x| x.0).collect();

        // -- baseline fixed
        let t0 = Instant::now();
        let (res, stats) = idx.search(q, K, baseline_fixed_ef);
        a_fixed.rec(recall_for(&res, &gt), stats.dists, baseline_fixed_ef, 0, t0.elapsed().as_micros());

        // -- baseline pessimist
        let t0 = Instant::now();
        let (res, stats) = idx.search(q, K, baseline_pessimist_ef);
        a_pess.rec(recall_for(&res, &gt), stats.dists, baseline_pessimist_ef, 0, t0.elapsed().as_micros());

        // -- oracle
        let t0 = Instant::now();
        let ef_o = OracleBudget::label(&idx, &corpus, DIM, q, &cfg, &ladder);
        let (res, stats) = idx.search(q, K, ef_o);
        a_oracle.rec(recall_for(&res, &gt), stats.dists, ef_o, 0, t0.elapsed().as_micros());

        // -- learned
        let t0 = Instant::now();
        let mut feat_stats = SearchStats::default();
        let ef_l = predictor.predict_query(q, &medoids, &idx, &mut feat_stats);
        let (res, stats) = idx.search(q, K, ef_l);
        a_learned.rec(
            recall_for(&res, &gt),
            stats.dists + feat_stats.dists,
            ef_l,
            feat_stats.dists,
            t0.elapsed().as_micros(),
        );
    }

    println!("\n--- results (k={}, target_recall={}) ---", K, TARGET_RECALL);
    a_fixed.print("baseline-fixed");
    a_pess.print("baseline-pessimist");
    a_oracle.print("oracle");
    a_learned.print("learned");

    let saved_vs_pess =
        1.0 - (a_learned.dist_sum as f64) / (a_pess.dist_sum as f64).max(1.0);
    let gap_vs_oracle =
        (a_learned.dist_sum as f64) / (a_oracle.dist_sum as f64).max(1.0);
    println!(
        "\nlearned vs baseline-pessimist : {:.1}% fewer distance computations",
        100.0 * saved_vs_pess
    );
    println!(
        "learned vs oracle              : {:.2}× distance cost (closer to 1.0 = better)",
        gap_vs_oracle
    );
    println!("\ntotal runtime: {} s", t_total.elapsed().as_secs_f64());

    let report = serde_json::json!({
        "config": {
            "dim": DIM,
            "n_corpus": N_CORPUS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "k": K,
            "target_recall": TARGET_RECALL,
            "hnsw_M": idx.params.m,
            "hnsw_ef_construction": idx.params.ef_construction,
            "n_medoids": N_MEDOIDS,
            "ef_ladder": ladder,
            "baseline_fixed_ef": baseline_fixed_ef,
            "baseline_pessimist_ef": baseline_pessimist_ef,
            "predictor_weights": predictor.w.to_vec(),
            "predictor_log2_margin": predictor.log2_margin,
        },
        "results": [
            a_fixed.json("baseline-fixed"),
            a_pess.json("baseline-pessimist"),
            a_oracle.json("oracle"),
            a_learned.json("learned"),
        ],
        "savings_vs_pessimist": saved_vs_pess,
        "ratio_vs_oracle": gap_vs_oracle,
    });
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("bench_results.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write bench_results.json");
    println!("wrote {}", out_path.display());
}
