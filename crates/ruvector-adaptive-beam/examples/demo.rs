//! Minimal demo: insert 2_000 points, run a query with each terminator.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_adaptive_beam::{
    FixedEfTerminator, Hnsw, HnswParams, QuantileTerminator, RatioTerminator,
};

fn main() {
    let dim = 48;
    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let data: Vec<Vec<f32>> = (0..2_000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let mut idx = Hnsw::new(dim, HnswParams::default());
    for v in &data {
        idx.insert(v);
    }
    let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    let mut fixed = FixedEfTerminator::new(64);
    let (r1, s1) = idx.search(&q, 10, &mut fixed);
    println!("fixed_ef:  top0={} exps={} dist={}", r1[0].id, s1.expansions, s1.distance_evals);

    let mut ratio = RatioTerminator::new(16, 1.10, 128);
    let (r2, s2) = idx.search(&q, 10, &mut ratio);
    println!("ratio:     top0={} exps={} dist={} early={}", r2[0].id, s2.expansions, s2.distance_evals, s2.early_stopped);

    let mut quant = QuantileTerminator::new(16, 0.75, 128);
    let (r3, s3) = idx.search(&q, 10, &mut quant);
    println!("quantile:  top0={} exps={} dist={} early={}", r3[0].id, s3.expansions, s3.distance_evals, s3.early_stopped);
}
