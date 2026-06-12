//! End-to-end demo: build HNSW on synthetic data, train the learned
//! predictor on a held-out query set, run all three policies on a test
//! set, and print recall + latency + distance-call counts.

use std::time::Instant;

use ruvector_early_term::data::synthesize;
use ruvector_early_term::{
    FixedEf, Hnsw, HnswParams, LearnedPolicy, RidgeRegressor, SearchStats, SlopePolicy,
};

fn recall_at_k(got: &[u32], gt: &[u32], k: usize) -> f32 {
    let k = k.min(got.len()).min(gt.len());
    let gt_set: std::collections::HashSet<u32> = gt[..k].iter().copied().collect();
    let mut hits = 0usize;
    for &g in &got[..k] { if gt_set.contains(&g) { hits += 1; } }
    hits as f32 / k as f32
}

fn run_policy(
    hnsw: &Hnsw,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
    label: &str,
    mut policy: Box<dyn ruvector_early_term::TerminationPolicy>,
) {
    let mut recalls = 0.0f32;
    let mut total_dc: u64 = 0;
    let mut total_exp: u64 = 0;
    let mut early_count: u64 = 0;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let (got, stats): (Vec<(f32, u32)>, SearchStats) =
            hnsw.search(q, k, ef, policy.as_mut());
        let ids: Vec<u32> = got.into_iter().map(|(_, i)| i).collect();
        recalls += recall_at_k(&ids, &gt[qi], k);
        total_dc += stats.distance_calls;
        total_exp += stats.expansions;
        if stats.terminated_early { early_count += 1; }
    }
    let elapsed = t0.elapsed();
    let nq = queries.len() as f32;
    println!(
        "[{:>14}] recall@{}={:.4}  dist_calls/q={:>6.1}  expansions/q={:>5.1}  early={:>4}/{}  total={:?}  per_query={:>6.1}us",
        label,
        k,
        recalls / nq,
        total_dc as f32 / nq,
        total_exp as f32 / nq,
        early_count,
        queries.len(),
        elapsed,
        elapsed.as_micros() as f32 / nq,
    );
}

fn main() {
    let n = 20_000;
    let nq_train = 200;
    let nq_test = 500;
    let dim = 64;
    let n_clusters = 64;
    let sigma = 0.18;
    let k = 10;
    let ef_max = 96;

    println!("== ruvector-early-term demo ==");
    println!(
        "n={}  dim={}  clusters={}  sigma={}  k={}  ef_max={}",
        n, dim, n_clusters, sigma, k, ef_max
    );

    println!("[step 1/4] synthesizing dataset + brute-force GT...");
    let ds = synthesize(n, nq_train + nq_test, dim, n_clusters, sigma, k, 42);
    let train_q = ds.queries[..nq_train].to_vec();
    let train_gt = ds.gt[..nq_train].to_vec();
    let test_q = ds.queries[nq_train..].to_vec();
    let test_gt = ds.gt[nq_train..].to_vec();

    println!("[step 2/4] building HNSW...");
    let mut hnsw = Hnsw::new(dim, HnswParams::default());
    let t0 = Instant::now();
    hnsw.build(ds.vectors);
    println!("           built in {:?}", t0.elapsed());

    println!("[step 3/4] training learned predictor on {} held-out queries...", nq_train);
    let predictor = train_predictor(&hnsw, &train_q, &train_gt, k, ef_max);

    println!("[step 4/4] evaluating policies on {} test queries...", nq_test);
    run_policy(
        &hnsw, &test_q, &test_gt, k, ef_max,
        "FixedEf",
        Box::new(FixedEf),
    );
    run_policy(
        &hnsw, &test_q, &test_gt, k, ef_max,
        "Slope(w=8,ε=2e-4)",
        Box::new(SlopePolicy::new(8, 2e-4)),
    );
    run_policy(
        &hnsw, &test_q, &test_gt, k, ef_max,
        "Learned(τ=0.04)",
        Box::new(LearnedPolicy::new(predictor.clone(), 0.04, 6)),
    );

    println!();
    println!("== Pareto sweep: recall@10 vs distance-calls/q ==");
    println!("[label                ] recall    dist_calls/q   us/q     early%");
    for ef in [32usize, 48, 64, 80, 96, 128] {
        sweep_run(&hnsw, &test_q, &test_gt, k, ef, "FixedEf",
            Box::new(FixedEf));
    }
    for eps in [5e-4f32, 2e-4, 1e-4, 5e-5] {
        sweep_run(&hnsw, &test_q, &test_gt, k, ef_max,
            &format!("Slope(ε={:.0e})", eps),
            Box::new(SlopePolicy::new(8, eps)));
    }
    for tau in [0.08f32, 0.06, 0.04, 0.02, 0.01] {
        sweep_run(&hnsw, &test_q, &test_gt, k, ef_max,
            &format!("Learned(τ={:.2})", tau),
            Box::new(LearnedPolicy::new(predictor.clone(), tau, 6)));
    }
}

fn sweep_run(
    hnsw: &Hnsw,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
    label: &str,
    mut policy: Box<dyn ruvector_early_term::TerminationPolicy>,
) {
    let mut recalls = 0.0f32;
    let mut total_dc: u64 = 0;
    let mut early: u64 = 0;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let (got, stats) = hnsw.search(q, k, ef, policy.as_mut());
        let ids: Vec<u32> = got.into_iter().map(|(_, i)| i).collect();
        recalls += recall_at_k(&ids, &gt[qi], k);
        total_dc += stats.distance_calls;
        if stats.terminated_early { early += 1; }
    }
    let el = t0.elapsed();
    let nq = queries.len() as f32;
    println!(
        "[{:>20}] {:.4}  {:>10.1}     {:>6.1}    {:>3.0}%",
        label,
        recalls / nq,
        total_dc as f32 / nq,
        el.as_micros() as f32 / nq,
        100.0 * early as f32 / nq,
    );
}

/// Train the predictor by replaying the FixedEf search on training queries and
/// emitting (features, residual-risk) pairs. The residual risk at step s is
/// (final_recall_at_ef_max - intermediate_recall_at_step_s). Predicting low
/// risk → safe to stop now.
fn train_predictor(
    hnsw: &Hnsw,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    ef_max: usize,
) -> RidgeRegressor {
    use ruvector_early_term::predictor::QueryFeatures;
    use ruvector_early_term::TerminationPolicy;

    struct Recorder {
        k: usize,
        ef_max: usize,
        rows: Vec<Vec<f32>>,
        per_query_rows: Vec<usize>,
        kth_history: Vec<f32>,
        cand_history: Vec<f32>,
    }
    impl TerminationPolicy for Recorder {
        fn reset(&mut self, k: usize, ef_max: usize) {
            self.k = k;
            self.ef_max = ef_max;
            self.kth_history.clear();
            self.cand_history.clear();
            self.per_query_rows.push(self.rows.len());
        }
        fn should_stop(&mut self, step: u32, cand: f32, kth: f32, top_len: usize) -> bool {
            if top_len < self.k || !kth.is_finite() { return false; }
            self.kth_history.push(kth);
            self.cand_history.push(cand);
            let feats = QueryFeatures::from_history(step, cand, kth, &self.kth_history, &self.cand_history, self.ef_max);
            self.rows.push(feats.to_vec());
            false
        }
    }

    let mut rec = Recorder {
        k, ef_max, rows: vec![], per_query_rows: vec![],
        kth_history: vec![], cand_history: vec![],
    };

    let mut final_recalls = Vec::with_capacity(queries.len());
    let mut intermediate_recalls_per_query: Vec<Vec<f32>> = Vec::with_capacity(queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let row_start = rec.rows.len();
        let (got, _) = hnsw.search(q, k, ef_max, &mut rec);
        let ids: Vec<u32> = got.into_iter().map(|(_, i)| i).collect();
        let final_r = recall_at_k(&ids, &gt[qi], k);
        final_recalls.push(final_r);
        // For each recorded row, we approximate intermediate recall by
        // proportional progress through the search; this is an inexpensive
        // but informative target signal.
        let row_end = rec.rows.len();
        let n_rows = (row_end - row_start).max(1);
        let mut per_q = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let progress = (i + 1) as f32 / n_rows as f32;
            // Sigmoid-shaped reach toward final_r: most recall is gained early.
            let reach = 1.0 - (-(3.5 * progress)).exp();
            per_q.push(final_r * reach);
        }
        intermediate_recalls_per_query.push(per_q);
        let _ = qi;
    }

    // Build (X, y) where y = residual risk = max(0, final - intermediate).
    let mut xs: Vec<Vec<f32>> = Vec::with_capacity(rec.rows.len());
    let mut ys: Vec<f32> = Vec::with_capacity(rec.rows.len());
    let mut cursor = 0usize;
    for (qi, per_q) in intermediate_recalls_per_query.iter().enumerate() {
        let final_r = final_recalls[qi];
        for (i, ir) in per_q.iter().enumerate() {
            xs.push(rec.rows[cursor + i].clone());
            ys.push((final_r - ir).max(0.0));
        }
        cursor += per_q.len();
    }

    RidgeRegressor::train(&xs, &ys, 1e-3)
}
