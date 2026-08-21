//! Minimal cascade example — build a Hamming coarse + FP32 rerank index
//! over 1k random vectors and print the top-5 for a single query.

use ruvector_hamming_cascade::{
    Cascade, CascadeConfig, Fp32Oracle, HammingOracle,
};

fn main() {
    let n = 1000;
    let dim = 64;
    let mut s: u64 = 0xFEED;
    let mut next = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let db: Vec<Vec<f32>> =
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
    let query: Vec<f32> = (0..dim).map(|_| next()).collect();

    let mut casc = Cascade::new(
        HammingOracle::from_vectors(&db),
        Fp32Oracle::from_vectors(&db),
        CascadeConfig { k: 5, probe_k: 50 },
    );
    for hit in casc.search(&query) {
        println!("id={:5}  score={:.4}", hit.id, hit.score);
    }
}
