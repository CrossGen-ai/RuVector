//! `finger-demo` — CLI reproduction of the numbers used in the ADR and gist.
//!
//! Runs three estimator variants (Exact, JL-r16, FINGER-r16) over the same
//! synthetic dataset and reports:
//!   * build time
//!   * per-query latency
//!   * recall@10 vs brute-force truth
//!   * bytes/vector for the approximation state
//!
//! No mocks: every number comes from `Instant::now()` around real code paths.

use ruvector_finger::estimator::DistanceEstimator;
use ruvector_finger::{brute_force_topk, recall_at_k, Dataset, ExactEstimator, FingerEstimator,
                       JlEstimator, PivotIndex};
use std::time::Instant;

fn top_k_by_approx<E: DistanceEstimator + ?Sized>(
    est: &E,
    idx: &PivotIndex,
    query: &[f32],
    beam: usize,
    rerank: usize,
    k: usize,
) -> (Vec<u32>, usize) {
    let entry = idx.entry(query, beam);
    let mut scored: Vec<(f32, u32)> = Vec::with_capacity(idx.vectors.len() / 4);
    for (pid, _) in &entry {
        let handle = est.prepare_query(query, *pid);
        for &nid in &idx.members[*pid as usize] {
            scored.push((handle.score(nid), nid));
        }
    }
    let candidates_seen = scored.len();
    // Keep top-`rerank` by approximate score.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(rerank);
    // Exact rerank of the shortlist.
    let mut exact_scored: Vec<(f32, u32)> = scored
        .into_iter()
        .map(|(_, id)| {
            let s: f32 = query.iter().zip(idx.vectors[id as usize].iter()).map(|(a, b)| a * b).sum();
            (s, id)
        })
        .collect();
    exact_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    (exact_scored.into_iter().take(k).map(|(_, id)| id).collect(), candidates_seen)
}

fn evaluate<E: DistanceEstimator + ?Sized>(
    label: &str,
    est: &E,
    idx: &PivotIndex,
    ds: &Dataset,
    truth: &[Vec<u32>],
    beam: usize,
    rerank: usize,
    k: usize,
) {
    let start = Instant::now();
    let mut recall_sum = 0f32;
    let mut candidates_total = 0usize;
    for (qi, q) in ds.queries.iter().enumerate() {
        let (pred, seen) = top_k_by_approx(est, idx, q, beam, rerank, k);
        recall_sum += recall_at_k(&pred, &truth[qi]);
        candidates_total += seen;
    }
    let elapsed = start.elapsed();
    let per_query_us = elapsed.as_secs_f64() * 1e6 / ds.queries.len() as f64;
    let recall = recall_sum / ds.queries.len() as f32;
    println!(
        "  {:<14} recall@{k}={:.3}  per-query={:>7.1}µs  bytes/vec={:>4}  avg_candidates={:>5}",
        label,
        recall,
        per_query_us,
        est.bytes_per_vector(),
        candidates_total / ds.queries.len()
    );
}

fn main() {
    let n = 20_000usize;
    let dim = 128usize;
    let n_queries = 200usize;
    let k = 10usize;
    let beam = 4usize;
    let rerank = 200usize;
    let n_pivots = (n as f64).sqrt() as usize; // ≈141

    println!("== ruvector-finger demo ==");
    println!("n={n}, d={dim}, queries={n_queries}, pivots={n_pivots}, beam={beam}, rerank={rerank}, k={k}");

    let t0 = Instant::now();
    let ds = Dataset::synthetic(n, n_queries, dim, 20260828);
    println!("dataset build: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let idx = PivotIndex::build(&ds, n_pivots, 1);
    println!("pivot index build: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = ds.queries.iter().map(|q| brute_force_topk(&ds, q, k)).collect();
    println!("brute-force truth: {:?}", t0.elapsed());

    let exact = ExactEstimator::new(&idx);
    let t0 = Instant::now();
    let jl16 = JlEstimator::build(&idx, 16, 42, "jl-r16");
    let build_jl = t0.elapsed();
    let t0 = Instant::now();
    let f16 = FingerEstimator::build(&idx, 16, 42, "finger-r16");
    let build_f16 = t0.elapsed();
    let t0 = Instant::now();
    let f8 = FingerEstimator::build(&idx, 8, 42, "finger-r8");
    let build_f8 = t0.elapsed();
    println!("estimator builds: jl-r16={:?}, finger-r16={:?}, finger-r8={:?}",
             build_jl, build_f16, build_f8);
    println!();
    println!("Search results (200 queries, beam={beam}, rerank={rerank}, k={k}):");

    evaluate("exact", &exact, &idx, &ds, &truth, beam, rerank, k);
    evaluate("jl-r16", &jl16, &idx, &ds, &truth, beam, rerank, k);
    evaluate("finger-r8", &f8, &idx, &ds, &truth, beam, rerank, k);
    evaluate("finger-r16", &f16, &idx, &ds, &truth, beam, rerank, k);

    println!();
    println!("(Numbers are single-threaded, release mode; see docs/research/nightly/…/README.md for methodology.)");
}
