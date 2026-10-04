use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use ruvector_batched_ivf::{
    batched_list_major_scan, batched_tiled_scan, build_ivf, serial_scan, QueryBatch,
};
use std::time::Instant;

fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

fn main() {
    let dim = 128;
    let n = 50_000;
    let n_lists = 200;
    let n_probe = 16;
    let k = 10;

    println!("Building IVF index: n={n} dim={dim} lists={n_lists} probe={n_probe} k={k}");
    let t0 = Instant::now();
    let vectors = gen(n, dim, 1);
    let index = build_ivf(vectors, dim, n_lists, 4, 2);
    println!("  built in {:?}", t0.elapsed());

    let avg_list = n as f32 / n_lists as f32;
    println!("  avg posting list size = {:.1}", avg_list);

    println!("\nBatch size | serial (ms) | list-major (ms) | tiled (ms) | speedup (tiled vs serial)");
    for &bs in &[1usize, 4, 16, 64, 256] {
        let queries = gen(bs, dim, 7);
        let batch = QueryBatch::new(queries, dim, k, n_probe);
        let probes = batch.probe(&index);

        let t = Instant::now();
        let a = serial_scan(&index, &batch, &probes);
        let ta = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let b = batched_list_major_scan(&index, &batch, &probes);
        let tb = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let c = batched_tiled_scan(&index, &batch, &probes);
        let tc = t.elapsed().as_secs_f64() * 1000.0;

        // Sanity: identical top-k
        for q in 0..batch.n {
            let ids_a: Vec<u32> = a[q].clone().into_sorted().into_iter().map(|p| p.1).collect();
            let ids_b: Vec<u32> = b[q].clone().into_sorted().into_iter().map(|p| p.1).collect();
            let ids_c: Vec<u32> = c[q].clone().into_sorted().into_iter().map(|p| p.1).collect();
            assert_eq!(ids_a, ids_b);
            assert_eq!(ids_a, ids_c);
        }

        println!("{:>10} | {:>11.2} | {:>15.2} | {:>10.2} | {:>7.2}×", bs, ta, tb, tc, ta / tc);
    }
}
