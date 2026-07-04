//! specann-demo — small runnable example.
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

fn main() {
    let n = 5_000;
    let dim = 128;
    let k = 10;
    println!("== specann-demo == n={} dim={} k={}", n, dim, k);

    let vecs = synth(n, dim, 42);
    let queries = synth(50, dim, 7);

    let baseline = F32BruteForce::from_vectors(&vecs).unwrap();

    // Ground truth (exact brute force).
    let truths: Vec<_> = queries
        .iter()
        .map(|q| {
            let mut r = DraftIndex::draft(&baseline, q, k).unwrap();
            r.sort();
            r
        })
        .collect();

    // Variant A: exact baseline recall (sanity).
    let mut ra = 0.0;
    for (q, t) in queries.iter().zip(&truths) {
        let r = DraftIndex::draft(&baseline, q, k).unwrap();
        ra += recall_at_k(&r, t, k);
    }
    println!("A  exact  recall@{}={:.3}", k, ra / queries.len() as f32);

    // Variant B: int8 draft only, no verification.
    let int8 = Int8BruteForce::from_vectors(&vecs).unwrap();
    let mut rb = 0.0;
    for (q, t) in queries.iter().zip(&truths) {
        let r = int8.draft(q, k).unwrap();
        rb += recall_at_k(&r, t, k);
    }
    println!("B  int8   recall@{}={:.3}  (no verify)", k, rb / queries.len() as f32);

    // Variant C: SpecANN (int8 draft + f32 verify + escalation).
    let draft = Int8BruteForce::from_vectors(&vecs).unwrap();
    let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
    let spec = SpecAnnIndex::new(draft, verifier, EscalationPolicy::default());
    let mut rc = 0.0;
    let mut total_verified = 0usize;
    for (q, t) in queries.iter().zip(&truths) {
        let (r, s) = spec.search(q, k).unwrap();
        rc += recall_at_k(&r, t, k);
        total_verified += s.verified;
    }
    println!(
        "C  spec   recall@{}={:.3}  avg-verified={:.1}",
        k,
        rc / queries.len() as f32,
        total_verified as f32 / queries.len() as f32
    );

    // Variant D: 1-bit draft + f32 verify (aggressive draft).
    let draft = Sign1BitDraft::from_vectors(&vecs).unwrap();
    let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
    let spec = SpecAnnIndex::new(
        draft,
        verifier,
        EscalationPolicy {
            alpha: 8,
            ..EscalationPolicy::default()
        },
    );
    let mut rd = 0.0;
    let mut total_verified = 0usize;
    for (q, t) in queries.iter().zip(&truths) {
        let (r, s) = spec.search(q, k).unwrap();
        rd += recall_at_k(&r, t, k);
        total_verified += s.verified;
    }
    println!(
        "D  1bit+f32 recall@{}={:.3}  avg-verified={:.1}",
        k,
        rd / queries.len() as f32,
        total_verified as f32 / queries.len() as f32
    );
}
