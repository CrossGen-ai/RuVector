//! adaptive-ef-demo: build a graph, sweep ef, train the predictor, and
//! report fixed_lo / fixed_hi / adaptive comparisons with real numbers.
//!
//! Run: `cargo run --release -p ruvector-adaptive-ef`
//! Reproducible — seed is fixed.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_adaptive_ef::adaptive::{extract_features, AdaptiveEf, EfFeatures};
use ruvector_adaptive_ef::nsw::{sq_l2, NswIndex, NswParams, SearchStats};

const SEED: u64 = 0xC0FFEE_u64;
const DIM: usize = 96;
const N: usize = 20_000;
const N_QUERY: usize = 1_000;
const N_TRAIN: usize = 400; // labelled probe set
const K: usize = 10;
const TARGET_RECALL: f64 = 0.95;
const EF_PROBE: usize = 16;
const EF_GRID: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

fn rand_unit(rng: &mut StdRng, d: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v { *x /= n; }
    v
}

/// Mixture of 16 isotropic Gaussians on the unit sphere — gives a wide
/// per-query difficulty spread so adaptive ef has something to chew on.
fn synth_mog(rng: &mut StdRng, n: usize, d: usize) -> Vec<Vec<f32>> {
    let n_centers = 16;
    let centers: Vec<Vec<f32>> = (0..n_centers).map(|_| rand_unit(rng, d)).collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..n_centers)];
            let mut v: Vec<f32> = c
                .iter()
                .map(|x| x + 0.35 * rng.gen_range(-1.0..1.0))
                .collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v { *x /= nrm; }
            v
        })
        .collect()
}

fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<u32> {
    let mut scored: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, sq_l2(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|x| x.0).collect()
}

fn recall(got: &[u32], gt: &HashSet<u32>) -> f64 {
    got.iter().filter(|i| gt.contains(*i)).count() as f64 / gt.len() as f64
}

/// For one query, find min ef on the grid that meets `target_recall`.
/// Returns `EF_GRID.last()` if none on grid achieves it.
fn label_query(idx: &NswIndex, q: &[f32], gt: &HashSet<u32>, target: f64) -> u32 {
    for &ef in EF_GRID {
        let (got, _) = idx.search(q, K, ef);
        if recall(&got, gt) + 1e-9 >= target {
            return ef as u32;
        }
    }
    *EF_GRID.last().unwrap() as u32
}

#[derive(Default, Clone, Copy)]
struct Summary {
    mean_dist: f64,
    p95_dist: f64,
    mean_recall: f64,
    p05_recall: f64,
    total_ns: u128,
    n: u64,
}

fn summarise(stats: &[(SearchStats, f64, u128)]) -> Summary {
    let mut dists: Vec<f64> = stats.iter().map(|s| s.0.distance_computations as f64).collect();
    let mut recs: Vec<f64> = stats.iter().map(|s| s.1).collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    recs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = stats.len();
    let p95_idx = (n as f64 * 0.95) as usize;
    let p05_idx = (n as f64 * 0.05) as usize;
    Summary {
        mean_dist: dists.iter().sum::<f64>() / n as f64,
        p95_dist: dists[p95_idx.min(n - 1)],
        mean_recall: recs.iter().sum::<f64>() / n as f64,
        p05_recall: recs[p05_idx.min(n - 1)],
        total_ns: stats.iter().map(|s| s.2).sum(),
        n: n as u64,
    }
}

fn print_summary(name: &str, s: Summary) {
    println!(
        "  {name:<10}  mean_dist={:>8.1}  p95_dist={:>8.1}  mean_recall={:.4}  p05_recall={:.4}  total_ms={:>7.1}",
        s.mean_dist,
        s.p95_dist,
        s.mean_recall,
        s.p05_recall,
        s.total_ns as f64 / 1.0e6
    );
}

fn run_fixed(idx: &NswIndex, queries: &[Vec<f32>], gts: &[HashSet<u32>], ef: usize) -> Summary {
    let mut rows = Vec::with_capacity(queries.len());
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t = Instant::now();
        let (got, st) = idx.search(q, K, ef);
        let dt = t.elapsed().as_nanos();
        rows.push((st, recall(&got, gt), dt));
    }
    summarise(&rows)
}

fn run_adaptive(
    idx: &NswIndex,
    model: &AdaptiveEf,
    queries: &[Vec<f32>],
    gts: &[HashSet<u32>],
) -> Summary {
    let mut rows = Vec::with_capacity(queries.len());
    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t = Instant::now();
        let feats = extract_features(idx, q, model.ef_probe as usize);
        let ef_pred = model.predict(&feats) as usize;
        let (got, mut st) = idx.search(q, K, ef_pred);
        // include probe work in the distance count so the comparison is fair
        st.distance_computations += feats.probe_stats.distance_computations;
        st.ef_used = ef_pred as u32;
        let dt = t.elapsed().as_nanos();
        rows.push((st, recall(&got, gt), dt));
    }
    summarise(&rows)
}

fn main() {
    println!("ruvector-adaptive-ef demo (seed={SEED}, dim={DIM}, n={N}, k={K})");
    let mut rng = StdRng::seed_from_u64(SEED);

    let t0 = Instant::now();
    let data = synth_mog(&mut rng, N, DIM);
    let queries = synth_mog(&mut rng, N_QUERY, DIM);
    println!(
        "generated data in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let mut idx = NswIndex::new(NswParams::new(DIM));
    for v in &data {
        idx.insert(v.clone());
    }
    println!(
        "built NSW({} nodes, M={}, ef_c={}) in {:.2}s",
        idx.len(),
        idx.params.m,
        idx.params.ef_construction,
        t0.elapsed().as_secs_f64()
    );

    println!("computing brute-force ground truth for {N_QUERY} queries...");
    let t0 = Instant::now();
    let gts: Vec<HashSet<u32>> = queries.iter().map(|q| brute_topk(&data, q, K)).collect();
    println!("brute-force GT in {:.2}s", t0.elapsed().as_secs_f64());

    // Labelling pass: first N_TRAIN queries
    println!("labelling first {N_TRAIN} queries (target_recall={TARGET_RECALL})...");
    let t0 = Instant::now();
    let mut train: Vec<(EfFeatures, u32)> = Vec::with_capacity(N_TRAIN);
    let mut min_ef_seen = u32::MAX;
    let mut max_ef_seen = 0;
    let mut p95_ef_label = 0;
    {
        let mut labels: Vec<u32> = Vec::with_capacity(N_TRAIN);
        for i in 0..N_TRAIN {
            let feats = extract_features(&idx, &queries[i], EF_PROBE);
            let label = label_query(&idx, &queries[i], &gts[i], TARGET_RECALL);
            min_ef_seen = min_ef_seen.min(label);
            max_ef_seen = max_ef_seen.max(label);
            labels.push(label);
            train.push((feats, label));
        }
        labels.sort();
        p95_ef_label = labels[((N_TRAIN as f64) * 0.95) as usize];
    }
    println!(
        "  ef_label range: [{min_ef_seen}, {max_ef_seen}], p95={p95_ef_label}, time={:.2}s",
        t0.elapsed().as_secs_f64()
    );

    // Fit predictor
    let ef_min = (*EF_GRID.first().unwrap()) as u32;
    let ef_max = (*EF_GRID.last().unwrap()) as u32;
    let mut model = AdaptiveEf::fit(&train, ef_min, ef_max, EF_PROBE as u32);
    println!("fit AdaptiveEf weights: {:?}", model.w);

    // Calibrate log_bias against the first 200 held-out queries to match
    // fixed_hi's mean recall. Calibration is a closure over the index/data.
    let calib_q = &queries[N_TRAIN..N_TRAIN + 200];
    let calib_gt = &gts[N_TRAIN..N_TRAIN + 200];
    let calib_target = TARGET_RECALL;
    let chosen_bias = model.calibrate_bias(calib_target, |m| {
        let mut rec_sum = 0.0;
        let mut work_sum = 0.0;
        for (q, gt) in calib_q.iter().zip(calib_gt.iter()) {
            let feats = extract_features(&idx, q, m.ef_probe as usize);
            let ef_pred = m.predict(&feats) as usize;
            let (got, st) = idx.search(q, K, ef_pred);
            rec_sum += recall(&got, gt);
            work_sum += (st.distance_computations + feats.probe_stats.distance_computations) as f64;
        }
        let n = calib_q.len() as f64;
        (rec_sum / n, work_sum / n)
    });
    println!("calibrated log_bias = {chosen_bias:.3} (multiplier {:.2}x)", 2.0f32.powf(chosen_bias));

    // Choose fixed baselines from training labels:
    //   * fixed_lo = median ef label  (under-shoots tail)
    //   * fixed_hi = p95 ef label     (textbook safe)
    let mut labels: Vec<u32> = train.iter().map(|t| t.1).collect();
    labels.sort();
    let median_ef = labels[labels.len() / 2] as usize;
    let p95_ef = p95_ef_label as usize;

    println!("baselines: fixed_lo (median ef) = {median_ef}, fixed_hi (p95 ef) = {p95_ef}");

    // Evaluate on the queries *not* used for training or calibration.
    let eval_start = N_TRAIN + 200;
    let eval_q = &queries[eval_start..];
    let eval_gt = &gts[eval_start..];
    println!("evaluating on {} held-out queries", eval_q.len());

    let s_lo = run_fixed(&idx, eval_q, eval_gt, median_ef);
    let s_hi = run_fixed(&idx, eval_q, eval_gt, p95_ef);
    let s_ad = run_adaptive(&idx, &model, eval_q, eval_gt);

    println!("\n=== RESULTS ===");
    print_summary("fixed_lo", s_lo);
    print_summary("fixed_hi", s_hi);
    print_summary("adaptive", s_ad);

    let speedup_mean = s_hi.mean_dist / s_ad.mean_dist;
    let speedup_p95 = s_hi.p95_dist / s_ad.p95_dist;
    println!(
        "\nAdaptive vs fixed_hi: mean distance-compute speedup = {:.2}x, p95 = {:.2}x",
        speedup_mean, speedup_p95
    );
    println!(
        "Recall: adaptive p05 = {:.4}  vs  fixed_hi p05 = {:.4}",
        s_ad.p05_recall, s_hi.p05_recall
    );
}
