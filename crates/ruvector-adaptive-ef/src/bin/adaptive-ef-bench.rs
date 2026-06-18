//! End-to-end benchmark: Fixed vs Heuristic vs Learned adaptive ef.
//!
//! What we measure:
//! * Mean **distance evaluations per query** (cost proxy — invariant to
//!   CPU noise, the right axis to publish nightly).
//! * Mean **wall-clock latency per query**, in microseconds.
//! * **Recall@k** against brute-force ground truth.
//!
//! What we compare (target recall = 0.90 at k=10):
//! 1. `FixedEf(ef)` for ef ∈ {16, 32, 64, 128, 256} — the classical
//!    sweep used to draw a Pareto front.
//! 2. `HeuristicAdaptiveEf(16, 256)` — LAET-style rule-based.
//! 3. `LearnedAdaptiveEf` — Auncel-style ridge regressor trained on a
//!    held-out fold.
//!
//! Dataset: synthetic mixture-of-Gaussians (10 clusters in R^64, 10 000
//! points), plus a 20% out-of-distribution query mix to stress-test the
//! adaptive predictors.  Synthetic data is the right call here because
//! we want reproducible nightly numbers; the algorithm is data-agnostic.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::time::Instant;

use ruvector_adaptive_ef::{
    extract_features, EfPredictor, FixedEf, HeuristicAdaptiveEf, LearnedAdaptiveEf, MiniHnsw,
    MiniHnswBuilder, TrainingSample,
};

const DIM: usize = 64;
const N_DATA: usize = 10_000;
const N_QUERIES: usize = 500;
const N_TRAIN: usize = 500;
const N_CLUSTERS: usize = 10;
const K: usize = 10;
const TARGET_RECALL: f32 = 0.90;
const OOD_FRACTION: f32 = 0.10;
const N_PIVOTS: usize = 8;

fn mog_sample(rng: &mut StdRng, centers: &[Vec<f32>], cluster_stds: &[f32]) -> Vec<f32> {
    use rand::Rng;
    let c = rng.gen_range(0..centers.len());
    let nd = Normal::new(0.0f32, cluster_stds[c]).unwrap();
    centers[c].iter().map(|x| x + nd.sample(rng)).collect()
}

fn ood_sample(rng: &mut StdRng) -> Vec<f32> {
    // OOD = far away from any cluster — sampled from a wide Gaussian
    // offset from the centroid hull.
    let nd = Normal::new(0.0f32, 8.0).unwrap();
    (0..DIM).map(|_| nd.sample(rng)).collect()
}

fn build_dataset(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let nd = Normal::new(0.0f32, 5.0).unwrap();
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| (0..DIM).map(|_| nd.sample(&mut rng)).collect())
        .collect();
    // Heterogeneous clusters: some tight (std=0.5), some loose (std=2.5).
    // Real-world datasets are like this — uniform spheres are unrealistic
    // and don't expose query-difficulty variation.
    let cluster_stds: Vec<f32> = (0..N_CLUSTERS)
        .map(|i| if i % 3 == 0 { 2.5 } else if i % 3 == 1 { 1.0 } else { 0.5 })
        .collect();
    let data: Vec<Vec<f32>> = (0..N_DATA).map(|_| mog_sample(&mut rng, &centers, &cluster_stds)).collect();

    // Pivots: random subset of data, fixed seed-derived choice.
    use rand::seq::SliceRandom;
    let mut idx: Vec<usize> = (0..N_DATA).collect();
    idx.shuffle(&mut rng);
    let pivots: Vec<Vec<f32>> = idx.iter().take(N_PIVOTS).map(|&i| data[i].clone()).collect();

    let mk = |n: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        let n_ood = ((n as f32) * OOD_FRACTION) as usize;
        let mut v: Vec<Vec<f32>> = (0..(n - n_ood))
            .map(|_| mog_sample(rng, &centers, &cluster_stds))
            .collect();
        for _ in 0..n_ood {
            v.push(ood_sample(rng));
        }
        v
    };

    let train_queries = mk(N_TRAIN, &mut rng);
    let test_queries = mk(N_QUERIES, &mut rng);
    (data, train_queries, test_queries, pivots)
}

fn recall_at_k(got: &[(u32, f32)], gt: &HashSet<u32>) -> f32 {
    let hits = got.iter().filter(|(id, _)| gt.contains(id)).count();
    hits as f32 / gt.len() as f32
}

/// Find the smallest ef in a candidate set that achieves `>= target` recall
/// for a single query.  Returns the max if none does.
fn min_ef_for_target(
    index: &MiniHnsw,
    q: &[f32],
    gt: &HashSet<u32>,
    candidates: &[usize],
    target: f32,
) -> usize {
    for &ef in candidates {
        let (got, _) = index.search(q, K, ef);
        let r = recall_at_k(&got, gt);
        if r >= target {
            return ef;
        }
    }
    *candidates.last().unwrap()
}

#[derive(Debug, Clone, Copy)]
struct Row {
    mean_de: f64,
    mean_lat_us: f64,
    recall: f64,
    p99_de: u64,
}

fn run_predictor<P: EfPredictor>(
    index: &MiniHnsw,
    queries: &[Vec<f32>],
    pivots: &[Vec<f32>],
    p: &P,
    ground_truth: &[HashSet<u32>],
) -> Row {
    let mut total_de: u64 = 0;
    let mut total_recall: f64 = 0.0;
    let mut total_us: f64 = 0.0;
    let mut des: Vec<u64> = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let f = extract_features(q, pivots);
        let ef = p.predict(&f);
        let t0 = Instant::now();
        let (got, stats) = index.search(q, K, ef);
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
        total_us += elapsed_us;
        total_de += stats.distance_evaluations;
        des.push(stats.distance_evaluations);
        total_recall += recall_at_k(&got, &ground_truth[qi]) as f64;
    }
    des.sort_unstable();
    let n = queries.len() as f64;
    Row {
        mean_de: total_de as f64 / n,
        mean_lat_us: total_us / n,
        recall: total_recall / n,
        p99_de: des[(des.len() as f64 * 0.99) as usize],
    }
}

fn main() {
    println!("== ruvector-adaptive-ef benchmark ==");
    println!(
        "dim={} N={} queries={} k={} target_recall={} ood_fraction={}",
        DIM, N_DATA, N_QUERIES, K, TARGET_RECALL, OOD_FRACTION
    );

    let t_build = Instant::now();
    let (data, train_queries, test_queries, pivots) = build_dataset(20260618);
    println!("dataset built in {:.2}s", t_build.elapsed().as_secs_f64());

    let t_idx = Instant::now();
    let index = MiniHnswBuilder::new(DIM)
        .m(16)
        .ef_construction(64)
        .seed(20260618)
        .build(data);
    println!("index built in {:.2}s", t_idx.elapsed().as_secs_f64());

    // Ground truth for all test queries.
    let t_gt = Instant::now();
    let ground_truth: Vec<HashSet<u32>> = test_queries
        .iter()
        .map(|q| index.brute_force(q, K).into_iter().map(|(i, _)| i).collect())
        .collect();
    println!("ground-truth (brute force) computed in {:.2}s", t_gt.elapsed().as_secs_f64());

    // Build training labels: for each training query, find min ef that
    // hits target recall.
    let t_train = Instant::now();
    let ef_grid: Vec<usize> = vec![8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];
    let train_samples: Vec<TrainingSample> = train_queries
        .iter()
        .map(|q| {
            let gt: HashSet<u32> = index.brute_force(q, K).into_iter().map(|(i, _)| i).collect();
            let ef = min_ef_for_target(&index, q, &gt, &ef_grid, TARGET_RECALL);
            let f = extract_features(q, &pivots);
            TrainingSample {
                features: f,
                min_ef_for_target: ef as f32,
            }
        })
        .collect();
    println!(
        "labels built in {:.2}s ({} samples)",
        t_train.elapsed().as_secs_f64(),
        train_samples.len()
    );

    let learned = LearnedAdaptiveEf::fit(&train_samples, 1e-2, 8, 512);
    println!("learned weights = {:?}", learned.weights());

    // -- Sweep fixed ef.
    let fixed_efs = [16usize, 32, 48, 64, 96, 128, 192, 256];
    println!("\n== Fixed ef sweep (Pareto baseline) ==");
    println!("ef\tmean_de\tp99_de\tmean_us\trecall");
    let mut fixed_rows: Vec<(usize, Row)> = Vec::new();
    for &ef in &fixed_efs {
        let p = FixedEf::new(ef);
        let r = run_predictor(&index, &test_queries, &pivots, &p, &ground_truth);
        println!(
            "{}\t{:.1}\t{}\t{:.1}\t{:.4}",
            ef, r.mean_de, r.p99_de, r.mean_lat_us, r.recall
        );
        fixed_rows.push((ef, r));
    }

    // -- Heuristic.
    println!("\n== Adaptive predictors ==");
    let heur = HeuristicAdaptiveEf::new(16, 256, 200.0);
    let r_heur = run_predictor(&index, &test_queries, &pivots, &heur, &ground_truth);
    println!(
        "Heuristic\tmean_de={:.1}\tp99_de={}\tmean_us={:.1}\trecall={:.4}",
        r_heur.mean_de, r_heur.p99_de, r_heur.mean_lat_us, r_heur.recall
    );

    // -- Learned.
    let r_lrn = run_predictor(&index, &test_queries, &pivots, &learned, &ground_truth);
    println!(
        "Learned\t\tmean_de={:.1}\tp99_de={}\tmean_us={:.1}\trecall={:.4}",
        r_lrn.mean_de, r_lrn.p99_de, r_lrn.mean_lat_us, r_lrn.recall
    );

    // -- Pareto summary: at recall >= TARGET, lowest mean_de wins.
    println!("\n== Pareto summary (recall >= {}) ==", TARGET_RECALL);
    let mut entries: Vec<(String, Row)> = fixed_rows
        .into_iter()
        .map(|(ef, r)| (format!("FixedEf({})", ef), r))
        .collect();
    entries.push(("HeuristicAdaptive".into(), r_heur));
    entries.push(("LearnedAdaptive".into(), r_lrn));
    entries.retain(|(_, r)| r.recall as f32 >= TARGET_RECALL);
    entries.sort_by(|a, b| a.1.mean_de.partial_cmp(&b.1.mean_de).unwrap());
    println!("rank\tstrategy\t\tmean_de\tp99_de\tmean_us\trecall");
    for (i, (n, r)) in entries.iter().enumerate() {
        println!(
            "{}\t{:<20}{:.1}\t{}\t{:.1}\t{:.4}",
            i + 1,
            n,
            r.mean_de,
            r.p99_de,
            r.mean_lat_us,
            r.recall
        );
    }
}
