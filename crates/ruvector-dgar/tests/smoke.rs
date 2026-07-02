//! End-to-end smoke test: PQ + all three rerankers over a small synthetic corpus.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_dgar::{
    AdaptiveGap, BruteForce, FixedK, OracleUpperBound, ProductQuantizer, Reranker,
};

fn gauss(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| d.sample(&mut rng)).collect())
        .collect()
}

#[test]
fn all_policies_return_k_and_track_recall() {
    let dim = 32;
    let corpus = gauss(2000, dim, 1);
    let queries = gauss(20, dim, 2);
    let pq = ProductQuantizer::train(&corpus, dim, 4, 32, 4, 3);
    let codes: Vec<Vec<u8>> = corpus.iter().map(|v| pq.encode(v)).collect();
    let exact = BruteForce { corpus: &corpus };
    let k = 5;

    for q in &queries {
        let approx = pq.search(q, &codes, 128);
        let truth: Vec<u32> = exact.topk(q, k).into_iter().map(|r| r.id).collect();

        let fk = FixedK { c: 8 };
        let (fk_out, fk_evals) = fk.rerank(q, &approx, k, &exact).unwrap();
        assert_eq!(fk_out.len(), k);
        assert!(fk_evals >= k);

        let ag = AdaptiveGap {
            gamma: 0.10,
            max_k: 16,
            min_k: 2,
        };
        let (ag_out, ag_evals) = ag.rerank(q, &approx, k, &exact).unwrap();
        assert_eq!(ag_out.len(), k);
        assert!(ag_evals >= 2 * k, "min_k floor must hold");
        assert!(ag_evals <= 16 * k, "max_k ceiling must hold");

        let oracle = OracleUpperBound { truth_ids: &truth };
        let (or_out, or_evals) = oracle.rerank(q, &approx, k, &exact).unwrap();
        assert_eq!(or_out.len(), k);
        assert_eq!(or_evals, k, "oracle must use exactly k exact evals");
        // Oracle has recall 1.0 by construction.
        let hits = or_out.iter().filter(|r| truth.contains(&r.id)).count();
        assert_eq!(hits, k);
    }
}

#[test]
fn adaptive_gap_never_beats_oracle_on_evals() {
    let dim = 16;
    let corpus = gauss(500, dim, 10);
    let queries = gauss(10, dim, 11);
    let pq = ProductQuantizer::train(&corpus, dim, 4, 16, 3, 12);
    let codes: Vec<Vec<u8>> = corpus.iter().map(|v| pq.encode(v)).collect();
    let exact = BruteForce { corpus: &corpus };
    let k = 5;

    let ag = AdaptiveGap {
        gamma: 0.05,
        max_k: 16,
        min_k: 2,
    };
    for q in &queries {
        let approx = pq.search(q, &codes, 128);
        let (_out, ag_evals) = ag.rerank(q, &approx, k, &exact).unwrap();
        // Oracle uses exactly k; adaptive must be >= k (it enforces a floor).
        assert!(ag_evals >= k);
    }
}

#[test]
fn error_when_k_exceeds_candidates() {
    let dim = 8;
    let corpus = gauss(100, dim, 20);
    let pq = ProductQuantizer::train(&corpus, dim, 2, 16, 2, 21);
    let codes: Vec<Vec<u8>> = corpus.iter().map(|v| pq.encode(v)).collect();
    let exact = BruteForce { corpus: &corpus };
    let approx = pq.search(&corpus[0], &codes, 10);
    let fk = FixedK { c: 4 };
    let res = fk.rerank(&corpus[0], &approx, 100, &exact);
    assert!(res.is_err());
}
