//! Integration test: recall on Gaussian corpus is monotone in bits.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use ruvector_extended_rabitq::{AnnIndex, ExtendedRabitqIndex, FlatF32Index};

fn gauss(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let v: f64 = rng.sample(StandardNormal);
                    v as f32
                })
                .collect()
        })
        .collect()
}

#[test]
fn recall_monotone_in_bits() {
    let dim = 64;
    let n = 2_000;
    let data = gauss(n, dim, 1);
    let queries = gauss(100, dim, 2);
    let flat = FlatF32Index::from_vectors(dim, &data).unwrap();
    let k = 10;
    let mut prev = 0.0f32;
    for &bits in &[1u32, 2, 4] {
        let idx = ExtendedRabitqIndex::build(dim, bits, 42, &data).unwrap();
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let gt: std::collections::HashSet<u32> =
                flat.search(q, k).unwrap().into_iter().map(|r| r.id).collect();
            let got = idx.search(q, k).unwrap();
            for r in &got {
                if gt.contains(&r.id) {
                    hits += 1;
                }
            }
            total += k;
        }
        let recall = hits as f32 / total as f32;
        println!("bits={bits} recall@10={recall:.3}");
        assert!(
            recall + 1e-3 >= prev,
            "recall regressed: bits={bits} recall={recall} prev={prev}"
        );
        prev = recall;
    }
    assert!(prev > 0.5, "4-bit recall too low: {prev}");
}

#[test]
fn deterministic_build() {
    let dim = 16;
    let data = gauss(64, dim, 3);
    let a = ExtendedRabitqIndex::build(dim, 4, 7, &data).unwrap();
    let b = ExtendedRabitqIndex::build(dim, 4, 7, &data).unwrap();
    let q = gauss(1, dim, 9).pop().unwrap();
    let ra = a.search(&q, 5).unwrap();
    let rb = b.search(&q, 5).unwrap();
    assert_eq!(ra, rb);
}
