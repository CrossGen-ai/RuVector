//! `opq-demo` — Train PQ / OPQ-NP / OPQ-P on a synthetic anisotropic dataset,
//! report reconstruction MSE and recall@10 against brute-force L2 ground truth.
//!
//! All numbers in the research doc are produced by this binary; no mocks.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_opq::recall::{ground_truth, recall_at_k};
use ruvector_opq::{mse, opq::{OpqNp, OpqP}, pq::Pq, Quantizer};
use std::time::Instant;

/// Synthetic anisotropic Gaussian-ish dataset. Variance decays exponentially
/// along axes — the regime where PQ's contiguous subspace partition is
/// maximally suboptimal and OPQ's rebalancing pays off.
///
/// All three quantizers see exactly the same data; the only thing that
/// changes is whether they pre-rotate before partitioning.
/// Synthetic distribution with smooth exponential variance decay across the
/// `d` axes. PQ's contiguous m-subspace partition is grossly unbalanced —
/// subspace 0 carries the high-variance axes, subspace m-1 carries near-zero
/// noise — so PQ wastes codebook capacity on the empty subspace and
/// under-resolves the high-variance one. OPQ's eigenvalue allocator
/// interleaves the axes so each subspace gets a balanced log-variance
/// product, the canonical setting Ge 2013 §6 reports gains on.
struct Synth {
    sigma: Vec<f32>,
    d: usize,
}

impl Synth {
    fn new(d: usize, decay: f32) -> Self {
        let sigma: Vec<f32> = (0..d).map(|j| (-(j as f32) * decay).exp()).collect();
        Self { sigma, d }
    }

    fn sample(&self, n: usize, seed: u64) -> Vec<f32> {
        let d = self.d;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let u: f32 = (0..4).map(|_| rng.gen::<f32>() - 0.5).sum();
                out[i * d + j] = u * self.sigma[j];
            }
        }
        out
    }
}

fn encode_all<Q: Quantizer>(q: &Q, data: &[f32], n: usize, d: usize) -> (Vec<u8>, Vec<f32>) {
    let m = q.m();
    let mut codes = vec![0u8; n * m];
    let mut recon = vec![0.0f32; n * d];
    let mut buf = vec![0.0f32; d];
    for i in 0..n {
        q.encode(&data[i * d..(i + 1) * d], &mut codes[i * m..(i + 1) * m]);
        q.decode(&codes[i * m..(i + 1) * m], &mut buf);
        recon[i * d..(i + 1) * d].copy_from_slice(&buf);
    }
    (codes, recon)
}

fn report<Q: Quantizer>(
    label: &str,
    q: &Q,
    base: &[f32],
    queries: &[f32],
    n_base: usize,
    n_query: usize,
    d: usize,
    gt: &[u32],
    k_recall: usize,
    train_ms: u128,
) {
    let t = Instant::now();
    let (codes, recon) = encode_all(q, base, n_base, d);
    let enc_ms = t.elapsed().as_millis();
    let m = mse(base, &recon, n_base, d);
    let t = Instant::now();
    let r = recall_at_k(q, &codes, n_base, queries, n_query, d, gt, k_recall);
    let scan_ms = t.elapsed().as_millis();
    println!(
        "  {:<10} train={:>5}ms  encode_base={:>4}ms  scan={:>5}ms  MSE={:.6}  recall@{}={:.3}",
        label, train_ms, enc_ms, scan_ms, m, k_recall, r
    );
}

fn run_regime(label: &str, d: usize, m: usize, decay: f32, n_train: usize, n_base: usize, n_query: usize) {
    let ds = d / m;
    let k_recall = 10;
    println!("\n=== {label} ===");
    println!("  d={d}  m={m}  ds={ds}  K=256  decay={decay}  n_train={n_train}  n_base={n_base}  n_query={n_query}");

    let synth = Synth::new(d, decay);
    let train = synth.sample(n_train, 1);
    let base = synth.sample(n_base, 2);
    let queries = synth.sample(n_query, 3);

    let t = Instant::now();
    let gt = ground_truth(&base, &queries, n_base, n_query, d, k_recall);
    println!("  ground-truth ({k_recall}-NN brute force): {} ms", t.elapsed().as_millis());

    let t = Instant::now();
    let mut pq = Pq::new(d, m);
    pq.fit(&train, n_train, d);
    let pq_train = t.elapsed().as_millis();
    report("PQ", &pq, &base, &queries, n_base, n_query, d, &gt, k_recall, pq_train);

    let t = Instant::now();
    let mut np = OpqNp::new(d, m);
    np.fit(&train, n_train, d);
    let np_train = t.elapsed().as_millis();
    report("OPQ-NP", &np, &base, &queries, n_base, n_query, d, &gt, k_recall, np_train);

    let iters = 4;
    let t = Instant::now();
    let mut p = OpqP::new(d, m, iters);
    p.fit(&train, n_train, d);
    let p_train = t.elapsed().as_millis();
    report(&format!("OPQ-P({iters})"), &p, &base, &queries, n_base, n_query, d, &gt, k_recall, p_train);

    println!("  Compression: {} bytes/vec vs raw {} bytes ({}x)", m, d * 4, (d * 4) / m);
}

fn main() {
    println!("ruvector-opq nightly benchmark — PQ vs OPQ-NP vs OPQ-P");
    // Regime A: wide subspaces (ds=16), moderate decay — classical OPQ setting where
    // K=256 centroids in 16-D get to exploit the rotation maximally.
    run_regime("A: d=64,  m=4  (ds=16) — wide subspaces, moderate decay", 64, 4, 0.04, 4_000, 10_000, 200);
    // Regime B: standard PQ width.
    run_regime("B: d=64,  m=8  (ds=8)  — standard PQ width",              64, 8, 0.04, 4_000, 10_000, 200);
    // Regime C: longer dimension, long-tail variance.
    run_regime("C: d=128, m=8  (ds=16) — long-tail decay",                128, 8, 0.05, 4_000, 10_000, 200);
}
