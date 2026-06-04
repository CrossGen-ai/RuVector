//! Tiny demo: encode + decode + score-by-lookup of 4 toy vectors.

use ruvector_anisotropic_pq::train::{train_anisotropic_pq, TrainOpts};
use ruvector_anisotropic_pq::{brute_force_topk, synthetic_unit_dataset};

fn main() {
    let n = 4_096;
    let dim = 32;
    let m = 4;
    let data = synthetic_unit_dataset(n, dim, 13);
    let pq = train_anisotropic_pq(&data, n, dim, m, 64, 4.0, TrainOpts { iters: 10, k: 64, seed: 1 });
    let codes = pq.encode(&data[0..dim]);
    let recon = pq.decode(&codes);
    println!("codes (first vector, {} bytes): {:?}", pq.code_size(), codes);
    println!("first 4 dims original:    {:?}", &data[0..4]);
    println!("first 4 dims reconstructed: {:?}", &recon[0..4]);

    // Score a query.
    let q = &data[1 * dim..2 * dim];
    let tbl = pq.build_lookup_ip(q);
    let codes_all = pq.encode_many(&data, n);
    let mut best = (0u32, f32::NEG_INFINITY);
    for i in 0..n {
        let est = pq.score_with_lookup(&tbl, &codes_all[i * m..(i + 1) * m]);
        if est > best.1 {
            best = (i as u32, est);
        }
    }
    let gt = brute_force_topk(&data, n, dim, q, 1)[0];
    println!("PQ top-1: idx={}, est_ip={:.4}", best.0, best.1);
    println!("true top-1: idx={}, true_ip={:.4}", gt.0, gt.1);
}
