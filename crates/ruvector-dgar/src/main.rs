//! DGAR benchmark binary.
//!
//! Runs three rerank policies (FixedK, AdaptiveGap, OracleUpperBound) against
//! a common PQ+brute-rerank pipeline on synthetic Gaussian-mixture data and
//! prints a real recall@10 vs. exact-eval-count comparison table.
//!
//! Invoke: `cargo run --release -p ruvector-dgar --bin dgar-bench`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_dgar::{
    AdaptiveGap, ApproxCandidate, BruteForce, FixedK, OracleUpperBound, ProductQuantizer, Reranker,
    RerankResult,
};
use std::time::Instant;

const N: usize = 20_000;
const DIM: usize = 128;
const N_QUERIES: usize = 200;
const K: usize = 10;
const M: usize = 16; // sub-quantizers -> dsub = 8
const KS: usize = 256;
const KMEANS_ITERS: usize = 10;
const N_APPROX: usize = 512; // approximate stage top-N feeding all rerankers
const SEED: u64 = 20260702;

fn gen_gaussian_mixture(n: usize, dim: usize, n_clusters: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centers = Vec::with_capacity(n_clusters);
    let cluster_spread = Normal::new(0.0f32, 4.0).unwrap();
    for _ in 0..n_clusters {
        let c: Vec<f32> = (0..dim).map(|_| cluster_spread.sample(rng)).collect();
        centers.push(c);
    }
    let noise = Normal::new(0.0f32, 1.0).unwrap();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let ci = rng.gen_range(0..n_clusters);
        let base = &centers[ci];
        let v: Vec<f32> = (0..dim).map(|j| base[j] + noise.sample(rng)).collect();
        out.push(v);
    }
    out
}

fn recall(pred: &[RerankResult], truth: &[u32]) -> f32 {
    let hits = pred
        .iter()
        .filter(|r| truth.contains(&r.id))
        .count() as f32;
    hits / truth.len() as f32
}

fn run_policy<R: Reranker>(
    label: &str,
    reranker: &R,
    queries: &[Vec<f32>],
    approx_lists: &[Vec<ApproxCandidate>],
    truths: &[Vec<u32>],
    exact: &BruteForce<'_>,
) {
    let mut total_evals = 0usize;
    let mut total_recall = 0.0f32;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let (pred, evals) = reranker
            .rerank(q, &approx_lists[qi], K, exact)
            .expect("rerank failed");
        total_evals += evals;
        total_recall += recall(&pred, &truths[qi]);
    }
    let dt = t0.elapsed().as_secs_f64();
    let avg_evals = total_evals as f32 / queries.len() as f32;
    let avg_recall = total_recall / queries.len() as f32;
    println!(
        "  {:<24}  avg_exact_evals={:7.2}  recall@{}={:.4}  wall_ms={:6.1}",
        label,
        avg_evals,
        K,
        avg_recall,
        dt * 1000.0
    );
}

fn main() {
    println!("=== DGAR benchmark ===");
    println!(
        "N={N} dim={DIM} queries={N_QUERIES} K={K} PQ(M={M},Ks={KS}) approx_top_N={N_APPROX} seed={SEED}"
    );
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("[1/4] generating Gaussian-mixture corpus + queries...");
    let corpus = gen_gaussian_mixture(N, DIM, 32, &mut rng);
    let queries = gen_gaussian_mixture(N_QUERIES, DIM, 32, &mut rng);

    println!("[2/4] training PQ (M={M}, Ks={KS}, iters={KMEANS_ITERS})...");
    let t = Instant::now();
    let pq = ProductQuantizer::train(&corpus, DIM, M, KS, KMEANS_ITERS, SEED);
    println!("      train wall = {:.2}s", t.elapsed().as_secs_f64());

    println!("[3/4] encoding corpus + computing approx top-N per query + ground truth...");
    let t = Instant::now();
    let corpus_codes: Vec<Vec<u8>> = corpus.iter().map(|v| pq.encode(v)).collect();
    println!("      encode wall = {:.2}s", t.elapsed().as_secs_f64());

    let exact = BruteForce { corpus: &corpus };
    let approx_lists: Vec<Vec<ApproxCandidate>> = queries
        .iter()
        .map(|q| pq.search(q, &corpus_codes, N_APPROX))
        .collect();
    let truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| exact.topk(q, K).into_iter().map(|r| r.id).collect())
        .collect();

    println!("[4/4] running rerankers...");

    for c in [2usize, 4, 8, 16, 32] {
        let r = FixedK { c };
        run_policy(&format!("FixedK  c={:>2}", c), &r, &queries, &approx_lists, &truths, &exact);
    }
    for gamma in [0.02f32, 0.05, 0.10, 0.20, 0.40] {
        let r = AdaptiveGap {
            gamma,
            max_k: 32,
            min_k: 2,
        };
        run_policy(
            &format!("AdaptiveGap γ={:.2}", gamma),
            &r,
            &queries,
            &approx_lists,
            &truths,
            &exact,
        );
    }
    for (qi, _) in queries.iter().enumerate().take(0) {
        let _ = qi;
    }
    // Oracle: use per-query truth as the "candidate set"
    let mut oracle_total_evals = 0usize;
    let mut oracle_total_recall = 0.0f32;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let r = OracleUpperBound {
            truth_ids: &truths[qi],
        };
        let (pred, evals) = r.rerank(q, &approx_lists[qi], K, &exact).unwrap();
        oracle_total_evals += evals;
        oracle_total_recall += recall(&pred, &truths[qi]);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "  {:<24}  avg_exact_evals={:7.2}  recall@{}={:.4}  wall_ms={:6.1}",
        "OracleUpperBound",
        oracle_total_evals as f32 / queries.len() as f32,
        K,
        oracle_total_recall / queries.len() as f32,
        dt * 1000.0
    );
    println!("=== done ===");
}
