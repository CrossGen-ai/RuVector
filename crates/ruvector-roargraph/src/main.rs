//! roargraph-demo — OOD ANNS benchmark: brute-force vs baseline vs RoarGraph.
//!
//! Generates:
//!   - N=5,000 base vectors in dim=64 from an 8-cluster Gaussian mixture (GMM A).
//!   - 500 training queries + 200 test queries from a *shifted* GMM (GMM B),
//!     modelling cross-modal OOD retrieval (e.g. CLIP text → image embeddings).
//!
//! Reports recall@10 and QPS for three strategies:
//!   1. Brute-force (ground truth, recall=1.0)
//!   2. Base-to-base k-NN graph (OOD-naive baseline)
//!   3. RoarGraph (bipartite projection over training queries)

use std::collections::HashSet;
use std::time::Instant;

use ruvector_roargraph::baseline::BaselineGraph;
use ruvector_roargraph::build::BuildParams;
use ruvector_roargraph::dataset::{exact_knn, generate_ood_dataset, DatasetParams};
use ruvector_roargraph::{AnnIndex, RoarGraphIndex};

// ─── configuration ────────────────────────────────────────────────────────────

const N_BASE: usize = 5_000;
const DIM: usize = 64;
const N_CLUSTERS: usize = 8;
const N_TRAIN_QUERIES: usize = 500;
const N_TEST_QUERIES: usize = 200;
const K: usize = 10;
const EF_SEARCH: usize = 50;
const MAX_DEGREE: usize = 32;
const K_TRAIN: usize = 20;
const SEED: u64 = 42;
const OOD_SHIFT: f32 = 3.0;

// ─── helpers ──────────────────────────────────────────────────────────────────

fn measure_recall(
    result_ids: &[usize],
    ground_truth: &HashSet<usize>,
) -> f64 {
    let hits = result_ids.iter().filter(|id| ground_truth.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}

fn run_brute_force(
    queries: &[Vec<f32>],
    base: &[Vec<f32>],
    ground_truth: &[HashSet<usize>],
) -> (f64, f64) {
    let t0 = Instant::now();
    let mut total_recall = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let result_set = exact_knn(q, base, K);
        let ids: Vec<usize> = result_set.iter().cloned().collect();
        total_recall += measure_recall(&ids, &ground_truth[qi]);
    }
    let elapsed = t0.elapsed();
    let recall = total_recall / queries.len() as f64;
    let qps = queries.len() as f64 / elapsed.as_secs_f64();
    (recall, qps)
}

fn run_index<Idx: AnnIndex>(
    idx: &Idx,
    queries: &[Vec<f32>],
    ground_truth: &[HashSet<usize>],
) -> (f64, f64, f64) {
    let t0 = Instant::now();
    let mut total_recall = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let results = idx.search(q, K, EF_SEARCH).expect("search failed");
        let ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        total_recall += measure_recall(&ids, &ground_truth[qi]);
    }
    let elapsed = t0.elapsed();
    let recall = total_recall / queries.len() as f64;
    let qps = queries.len() as f64 / elapsed.as_secs_f64();
    let mean_us = elapsed.as_micros() as f64 / queries.len() as f64;
    (recall, qps, mean_us)
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("ruvector-roargraph OOD ANNS demo");
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "config: N={N_BASE}  dim={DIM}  clusters={N_CLUSTERS}  \
         ood_shift={OOD_SHIFT}  train_q={N_TRAIN_QUERIES}  \
         test_q={N_TEST_QUERIES}  K={K}  ef={EF_SEARCH}  \
         max_degree={MAX_DEGREE}  k_train={K_TRAIN}"
    );

    // Generate OOD dataset
    println!("\nGenerating OOD dataset …");
    let params = DatasetParams {
        n_base: N_BASE,
        dim: DIM,
        n_clusters: N_CLUSTERS,
        cluster_std: 0.5,
        cluster_range: 5.0,
        ood_shift: OOD_SHIFT,
        seed: SEED,
    };
    let t_gen = Instant::now();
    let ds = generate_ood_dataset(&params, N_TRAIN_QUERIES, N_TEST_QUERIES, K);
    println!("  dataset generated in {:.1}ms", t_gen.elapsed().as_millis());
    println!(
        "  base={} train_queries={} test_queries={} gt_sets={}",
        ds.base.len(),
        ds.train_queries.len(),
        ds.test_queries.len(),
        ds.ground_truth.len()
    );

    // ── Variant 1: Brute-force ────────────────────────────────────────────────
    println!("\n── 1. Brute-force (exact, recall reference) ──");
    let t_bf = Instant::now();
    let (bf_recall, bf_qps) = run_brute_force(&ds.test_queries, &ds.base, &ds.ground_truth);
    let bf_us = t_bf.elapsed().as_micros() as f64 / N_TEST_QUERIES as f64;
    println!(
        "  recall@{K}: {:.1}%  |  QPS: {:.0}  |  mean latency: {:.1} µs",
        bf_recall * 100.0,
        bf_qps,
        bf_us
    );

    // ── Variant 2: Baseline (base-to-base k-NN graph) ─────────────────────────
    println!("\n── 2. Baseline: base-to-base k-NN graph (OOD-naive) ──");
    let mut baseline = BaselineGraph::new(DIM, MAX_DEGREE);
    baseline.add(&ds.base).unwrap();
    print!("  building … ");
    let t_build = Instant::now();
    baseline.build(&[]).unwrap();
    let build_ms = t_build.elapsed().as_millis();
    println!("done in {build_ms}ms");
    let (bl_recall, bl_qps, bl_us) =
        run_index(&baseline, &ds.test_queries, &ds.ground_truth);
    println!(
        "  recall@{K}: {:.1}%  |  QPS: {:.0}  |  mean latency: {:.1} µs  |  build: {build_ms}ms",
        bl_recall * 100.0,
        bl_qps,
        bl_us,
    );

    // ── Variant 3: RoarGraph ──────────────────────────────────────────────────
    println!("\n── 3. RoarGraph (bipartite projection, query-aware) ──");
    let roar_params = BuildParams {
        k_train: K_TRAIN,
        max_degree: MAX_DEGREE,
    };
    let mut roar = RoarGraphIndex::new(DIM, roar_params);
    roar.add(&ds.base).unwrap();
    print!("  building (using {} training queries) … ", N_TRAIN_QUERIES);
    let t_build_roar = Instant::now();
    roar.build(&ds.train_queries).unwrap();
    let roar_build_ms = t_build_roar.elapsed().as_millis();
    println!("done in {roar_build_ms}ms");
    let (rg_recall, rg_qps, rg_us) =
        run_index(&roar, &ds.test_queries, &ds.ground_truth);
    println!(
        "  recall@{K}: {:.1}%  |  QPS: {:.0}  |  mean latency: {:.1} µs  |  build: {roar_build_ms}ms",
        rg_recall * 100.0,
        rg_qps,
        rg_us,
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!(
        "{:<40} {:>10} {:>12} {:>14} {:>12}",
        "Variant", "recall@10", "mean µs", "QPS", "build ms"
    );
    println!("{}", "─".repeat(90));
    println!(
        "{:<40} {:>9.1}% {:>12.1} {:>14.0} {:>12}",
        "Brute-force (exact)", bf_recall * 100.0, bf_us, bf_qps, "—"
    );
    println!(
        "{:<40} {:>9.1}% {:>12.1} {:>14.0} {:>12}",
        "Baseline (base-to-base k-NN)", bl_recall * 100.0, bl_us, bl_qps, build_ms
    );
    println!(
        "{:<40} {:>9.1}% {:>12.1} {:>14.0} {:>12}",
        "RoarGraph (bipartite projection)", rg_recall * 100.0, rg_us, rg_qps, roar_build_ms
    );
    println!();

    // Delta reporting
    let delta_pp = (rg_recall - bl_recall) * 100.0;
    let sign = if delta_pp >= 0.0 { "+" } else { "" };
    println!(
        "RoarGraph vs baseline: {sign}{delta_pp:.1} pp recall@{K} improvement"
    );
    println!("(higher OOD shift → greater RoarGraph advantage)");
    println!();
}
