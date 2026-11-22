//! Quick demo: encode 2000 random vectors with GCSQ, Binary SQ, and SimHash,
//! and print quality numbers for a 50-query probe.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_gcsq::backends::{evaluate, BinarySqReranker, GcsqReranker, SimHashReranker};

fn make(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let u1: f32 = rng.gen_range(1e-9f32..1.0);
                    let u2: f32 = rng.gen_range(0.0f32..1.0);
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
                })
                .collect()
        })
        .collect()
}

fn main() {
    let dim = 64;
    let base = make(2000, dim, 1);
    let queries = make(50, dim, 2);

    let gc = GcsqReranker::fit(&base, 2);
    let bn = BinarySqReranker::fit(&base, 2);
    let sh = SimHashReranker::fit(dim, dim * 2, 99);

    for r in [
        evaluate(&gc, &base, &queries, 10),
        evaluate(&bn, &base, &queries, 10),
        evaluate(&sh, &base, &queries, 10),
    ] {
        println!(
            "{:<10} code_bytes={:>3} spearman={:.4} recall@10={:.4}",
            r.name, r.code_bytes, r.spearman_rho, r.recall_at_k
        );
    }
}
