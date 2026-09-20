//! Smoke test: verify recall parity between baseline and early-abandon.
use ruvector_partial_l2::{rerank_early_abandon_dyn, rerank_full};

fn make(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let dim = 768;
    let n = 1000;
    let k = 10;
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make(1000 + i as u64, dim)).collect();
    let q = make(1, dim);
    let cands: Vec<u32> = (0..n as u32).collect();

    let full = rerank_full(&q, &vectors, &cands, k);
    let ea = rerank_early_abandon_dyn(&q, &vectors, &cands, k, 64);
    let full_ids: Vec<u32> = full.iter().map(|x| x.1).collect();
    let ea_ids: Vec<u32> = ea.iter().map(|x| x.1).collect();
    assert_eq!(full_ids, ea_ids);
    println!("smoke ok — top-{k} ids match, tau_sq={:.4}", full.last().unwrap().0);
}
