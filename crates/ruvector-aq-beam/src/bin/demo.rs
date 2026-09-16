//! aq-beam-demo: measures PQ vs AQ (beam=1/8/32) on a synthetic
//! low-rank + noise dataset. Reports encode time, reconstruction MSE,
//! search latency, and recall@10 versus exhaustive f32 ground truth.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_aq_beam::{
    brute_force_topk, recall_at_k, AqBeamIndex, AqParams, PqIndex, PqParams, Quantizer,
};

fn gen_dataset(n: usize, d: usize, rank: usize, noise: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    // Random basis of `rank` orthonormal-ish directions in R^d.
    let basis: Vec<Vec<f32>> = (0..rank)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect();
    let eps = Normal::new(0.0f32, noise).unwrap();
    (0..n)
        .map(|_| {
            let coeffs: Vec<f32> = (0..rank).map(|_| normal.sample(&mut rng)).collect();
            let mut v = vec![0.0f32; d];
            for (r, c) in coeffs.iter().enumerate() {
                for j in 0..d {
                    v[j] += c * basis[r][j];
                }
            }
            for j in 0..d {
                v[j] += eps.sample(&mut rng);
            }
            v
        })
        .collect()
}

fn mean_mse<Q: Quantizer + ?Sized>(q: &Q, xs: &[Vec<f32>], codes: &[Vec<u8>]) -> f32 {
    let mut acc = 0.0f64;
    for (i, code) in codes.iter().enumerate() {
        let r = q.reconstruct(code);
        for j in 0..xs[i].len() {
            let dd = xs[i][j] - r[j];
            acc += (dd * dd) as f64;
        }
    }
    (acc / (codes.len() as f64 * xs[0].len() as f64)) as f32
}

fn run_variant(
    name: &str,
    q: &(dyn Quantizer),
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<(usize, f32)>],
    k: usize,
) {
    let t0 = Instant::now();
    let codes = q.encode_dataset(base);
    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mse = mean_mse(q, base, &codes);

    let mut recall_sum = 0.0f32;
    let t1 = Instant::now();
    for (qi, qv) in queries.iter().enumerate() {
        let got = q.search(qv, &codes, k);
        recall_sum += recall_at_k(&got, &truth[qi]);
    }
    let search_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let recall = recall_sum / queries.len() as f32;

    println!(
        "{name:<24} code_bytes={cb:>3}  encode={enc:>9.1}ms  MSE={mse:>10.5}  search_total={s:>7.1}ms  recall@{k}={r:.3}",
        cb = q.code_bytes(),
        enc = encode_ms,
        mse = mse,
        s = search_ms,
        k = k,
        r = recall
    );
}

fn main() {
    // Reasonably small numbers so the demo finishes in seconds.
    let n = 8_000usize;
    let d = 128usize;
    let n_query = 200usize;
    let k = 10usize;
    let rank = 24usize;
    let noise = 0.15f32;

    println!("aq-beam-demo: n={n} d={d} queries={n_query} k={k} rank={rank} noise={noise}");
    let data = gen_dataset(n, d, rank, noise, 42);
    let queries = gen_dataset(n_query, d, rank, noise, 43);

    // Exhaustive f32 ground truth.
    let t0 = Instant::now();
    let truth: Vec<Vec<(usize, f32)>> = queries.iter().map(|q| brute_force_topk(q, &data, k)).collect();
    println!(
        "ground truth built in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Baseline PQ: M=4, K=256 (32-bit codes, same footprint as the AQ variants).
    let pq = PqIndex::train(
        &data,
        PqParams { m: 4, k: 256, kmeans_iters: 12, seed: 0xC0FFEE },
    )
    .expect("pq train");
    run_variant("PQ (M=4, K=256)", &pq, &data, &queries, &truth, k);

    for beam in [1, 8, 32] {
        let aq = AqBeamIndex::train(
            &data,
            AqParams { m: 4, k: 256, beam, kmeans_iters: 12, seed: 0xBEEF },
        )
        .expect("aq train");
        let label = format!("AQ M=4 K=256 beam={beam}");
        run_variant(&label, &aq, &data, &queries, &truth, k);
    }

    println!("done.");
}
