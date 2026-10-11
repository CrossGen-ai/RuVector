//! Hand-rolled benchmark (no criterion dep): encode throughput + scan latency
//! for each bit width. Prints real numbers consumed by the research doc.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use ruvector_extended_rabitq::{quantize::Index, scan::topk_l2, ExtendedRaBitQ};
use std::time::Instant;

fn gen_vec(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

fn main() {
    let dims = [128usize, 256, 512];
    let n = 10_000;
    let n_queries = 500;
    println!(
        "{:<5} {:<5} {:<12} {:<14} {:<14}",
        "dim", "bits", "enc_vec/s", "scan_us/vec", "scan_Mvec/s"
    );
    for &dim in &dims {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let base: Vec<Vec<f32>> = (0..n).map(|_| gen_vec(&mut rng, dim)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_vec(&mut rng, dim)).collect();
        for bits in [1u8, 2, 4, 8] {
            let q = ExtendedRaBitQ::new(dim, bits, 1).unwrap();
            let t0 = Instant::now();
            let mut idx = Index::new(q);
            idx.add(&base).unwrap();
            let enc_s = n as f32 / t0.elapsed().as_secs_f32();

            let t0 = Instant::now();
            let mut sink = 0.0f32;
            for qv in &queries {
                let r = topk_l2(&idx, qv, 10);
                sink += r[0].1;
            }
            let elapsed = t0.elapsed().as_secs_f32();
            let scan_vecs = (n * n_queries) as f32;
            let per_vec_us = elapsed / scan_vecs * 1e6;
            let mvec_s = scan_vecs / elapsed / 1e6;
            // Prevent dead code elimination.
            if !sink.is_finite() {
                println!("sink NaN");
            }
            println!(
                "{:<5} {:<5} {:<12.1} {:<14.4} {:<14.2}",
                dim, bits, enc_s, per_vec_us, mvec_s
            );
        }
    }
}
