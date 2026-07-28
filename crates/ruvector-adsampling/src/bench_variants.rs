//! Real (non-mocked) end-to-end variant harness.
//!
//! Given a corpus, a query workload, and a top-k, produce a report per
//! oracle variant with:
//! * throughput (queries/s)
//! * scalar-op count per query (the paper's headline metric)
//! * recall@k against the exact answer

use crate::index::{AdsIndex, Neighbor};
use crate::oracle::{AdsAdaptive, AdsFixedBudget, DistanceOracle, ExactL2};

/// One row of the report table.
#[derive(Debug, Clone)]
pub struct VariantReport {
    /// Oracle identifier.
    pub name: &'static str,
    /// Wall-clock seconds spent searching all queries.
    pub elapsed_s: f64,
    /// Queries per second.
    pub qps: f64,
    /// Mean scalar float ops per query.
    pub ops_per_query: f64,
    /// Mean recall@k against exact answer.
    pub recall_at_k: f64,
    /// Fraction of candidate distance calls the oracle pruned early.
    pub prune_rate: f64,
}

/// Build a rotated index over `corpus`, run each of the three built-in
/// oracles against `queries`, and return a report per variant.
pub fn bench_variants(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    seed: u64,
) -> Vec<VariantReport> {
    let idx = AdsIndex::build(corpus, seed).expect("build");
    let d = idx.dim();

    // Ground truth via exact search (single pass, reused).
    let exact_oracle = ExactL2::new();
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(queries.len());
    for q in queries {
        let r: Vec<Neighbor> = idx.search(q, k, &exact_oracle).unwrap();
        truth.push(r.iter().map(|n| n.id).collect());
    }
    // Reset stats for the actual reported ExactL2 row.
    let exact_oracle = ExactL2::new();

    let variants: Vec<Box<dyn DistanceOracle>> = vec![
        Box::new(exact_oracle),
        Box::new(AdsFixedBudget::new(d / 4)),
        Box::new(AdsAdaptive::with_epsilon_from_dim(d)),
    ];

    let mut out = Vec::with_capacity(variants.len());
    for v in variants {
        let name = v.name();
        let start = std::time::Instant::now();
        let mut recall_sum = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let r = idx.search(q, k, v.as_ref()).unwrap();
            let got: std::collections::HashSet<u32> = r.iter().map(|n| n.id).collect();
            let hit = truth[qi].iter().filter(|id| got.contains(id)).count();
            recall_sum += hit as f64 / k as f64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let stats = v.stats();
        let ops_per_query = stats.scalar_ops as f64 / queries.len() as f64;
        let prune_rate = if stats.evals == 0 {
            0.0
        } else {
            stats.pruned as f64 / stats.evals as f64
        };
        out.push(VariantReport {
            name,
            elapsed_s: elapsed,
            qps: queries.len() as f64 / elapsed.max(1e-9),
            ops_per_query,
            recall_at_k: recall_sum / queries.len() as f64,
            prune_rate,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| r.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    #[test]
    fn variant_report_shape_is_correct() {
        let corpus = synth(1024, 96, 1);
        let queries = synth(32, 96, 2);
        let rows = bench_variants(&corpus, &queries, 10, 42);
        assert_eq!(rows.len(), 3);
        // Every variant must have positive ops and finite qps.
        for r in &rows {
            assert!(r.ops_per_query > 0.0, "{}", r.name);
            assert!(r.qps.is_finite() && r.qps > 0.0, "{}", r.name);
            assert!(r.recall_at_k >= 0.0 && r.recall_at_k <= 1.0);
        }
        // Exact must be recall = 1.0.
        let exact = rows.iter().find(|r| r.name == "exact-l2").unwrap();
        assert!((exact.recall_at_k - 1.0).abs() < 1e-9);
    }

    #[test]
    fn adaptive_uses_fewer_ops_than_exact() {
        let corpus = synth(2048, 128, 3);
        let queries = synth(64, 128, 4);
        let rows = bench_variants(&corpus, &queries, 10, 42);
        let exact = rows.iter().find(|r| r.name == "exact-l2").unwrap();
        let ada = rows.iter().find(|r| r.name == "ads-adaptive").unwrap();
        assert!(
            ada.ops_per_query < exact.ops_per_query,
            "ads-adaptive ops_per_query={} not less than exact-l2 ops_per_query={}",
            ada.ops_per_query,
            exact.ops_per_query
        );
    }
}
