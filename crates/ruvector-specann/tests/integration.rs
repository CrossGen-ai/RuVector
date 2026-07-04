//! End-to-end acceptance tests for ruvector-specann.
use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_specann::{
    recall_at_k, DraftIndex, EscalationPolicy, F32BruteForce, Int8BruteForce, Sign1BitDraft,
    SpecAnnIndex,
};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.sample(StandardNormal)).collect())
        .collect()
}

/// Numeric acceptance: SpecANN with int8 draft + f32 verifier must match exact
/// float32 baseline at ≥ 0.98 recall@10 on a synthetic gaussian corpus.
#[test]
fn acceptance_int8_specann_recall_ge_098() {
    let n = 2_000;
    let dim = 96;
    let k = 10;
    let vecs = synth(n, dim, 123);
    let queries = synth(50, dim, 456);

    let baseline = F32BruteForce::from_vectors(&vecs).unwrap();
    let truths: Vec<_> = queries
        .iter()
        .map(|q| {
            let mut r = DraftIndex::draft(&baseline, q, k).unwrap();
            r.sort();
            r
        })
        .collect();

    let draft = Int8BruteForce::from_vectors(&vecs).unwrap();
    let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
    let spec = SpecAnnIndex::new(draft, verifier, EscalationPolicy::default());

    let mut acc = 0.0f32;
    for (q, t) in queries.iter().zip(&truths) {
        let (r, _stats) = spec.search(q, k).unwrap();
        acc += recall_at_k(&r, t, k);
    }
    let avg = acc / queries.len() as f32;
    assert!(
        avg >= 0.98,
        "int8 SpecANN recall@10 = {:.3}, expected >= 0.98",
        avg
    );
}

/// The 1-bit variant is coarse and MUST rely on escalation to hit ≥ 0.90.
#[test]
fn acceptance_1bit_specann_recall_ge_090_with_escalation() {
    let n = 2_000;
    let dim = 128;
    let k = 10;
    let vecs = synth(n, dim, 7);
    let queries = synth(50, dim, 8);

    let baseline = F32BruteForce::from_vectors(&vecs).unwrap();
    let truths: Vec<_> = queries
        .iter()
        .map(|q| {
            let mut r = DraftIndex::draft(&baseline, q, k).unwrap();
            r.sort();
            r
        })
        .collect();

    let draft = Sign1BitDraft::from_vectors(&vecs).unwrap();
    let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
    let spec = SpecAnnIndex::new(
        draft,
        verifier,
        EscalationPolicy {
            alpha: 12,
            gap_threshold: 0.05,
            max_escalations: 2,
            escalate_multiplier: 2.5,
        },
    );

    let mut acc = 0.0f32;
    for (q, t) in queries.iter().zip(&truths) {
        let (r, _s) = spec.search(q, k).unwrap();
        acc += recall_at_k(&r, t, k);
    }
    let avg = acc / queries.len() as f32;
    assert!(
        avg >= 0.90,
        "1-bit SpecANN recall@10 = {:.3}, expected >= 0.90",
        avg
    );
}
