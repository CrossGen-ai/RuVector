//! End-to-end demo: train FASQ + baselines on synthetic anisotropic data,
//! print MSE, storage, and recall@10 against a brute-force float ground truth.

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_fasq::{
    baseline::{UniformSq4, UniformSq8},
    quantizer::Fasq,
    recon_mse, Quantizer,
};
use std::time::Instant;

fn gen_anisotropic(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Per-dim stddev: geometric decay -> high anisotropy.
    let stds: Vec<f32> = (0..d).map(|i| 4.0 * 0.85f32.powi(i as i32) + 0.05).collect();
    let normals: Vec<Normal<f32>> = stds.iter().map(|s| Normal::new(0.0, *s).unwrap()).collect();
    (0..n)
        .map(|_| (0..d).map(|i| normals[i].sample(&mut rng)).collect())
        .collect()
}

fn gen_isotropic(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..n).map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect()
}

fn brute_force_knn(base: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut d = 0.0f32;
            for j in 0..v.len() {
                let e = v[j] - q[j];
                d += e * e;
            }
            (d, i)
        })
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(k).map(|(_, i)| i).collect()
}

fn quantized_knn<Q: Quantizer>(q: &Q, codes: &[Vec<u8>], base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scratch = Vec::new();
    let mut dists: Vec<(f32, usize)> = codes.iter().enumerate().map(|(i, c)| {
        let d = q.distance_sq(query, c, &mut scratch).unwrap();
        (d, i)
    }).collect();
    let _ = base; // signature parity
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall_at_k(gt: &[usize], approx: &[usize]) -> f32 {
    let hits = approx.iter().filter(|i| gt.contains(i)).count();
    hits as f32 / gt.len() as f32
}

fn run(n_train: usize, n_base: usize, n_query: usize, d: usize, seed: u64, label: &str, generator: fn(usize, usize, u64) -> Vec<Vec<f32>>) {
    println!("\n=== {label} :: dim={d} train={n_train} base={n_base} q={n_query} ===");
    let train = generator(n_train, d, seed);
    let base = generator(n_base, d, seed + 1);
    let queries = generator(n_query, d, seed + 2);

    // Train.
    let t0 = Instant::now();
    let sq8 = UniformSq8::train(&train).unwrap();
    let dur_sq8_train = t0.elapsed();
    let t0 = Instant::now();
    let sq4 = UniformSq4::train(&train).unwrap();
    let dur_sq4_train = t0.elapsed();
    let t0 = Instant::now();
    let fasq = Fasq::train(&train, 4.0, 2, 8).unwrap();
    let dur_fasq_train = t0.elapsed();

    println!(
        "storage bytes/vec  SQ8={}  SQ4={}  FASQ={}  (FASQ avg bits/dim = {:.3})",
        sq8.bytes_per_vector(),
        sq4.bytes_per_vector(),
        fasq.bytes_per_vector(),
        fasq.total_bits as f32 / d as f32,
    );
    println!("train time         SQ8={:.2?}  SQ4={:.2?}  FASQ={:.2?}",
        dur_sq8_train, dur_sq4_train, dur_fasq_train);

    // Reconstruction MSE on held-out queries.
    let mse_sq8 = recon_mse(&sq8, &queries).unwrap();
    let mse_sq4 = recon_mse(&sq4, &queries).unwrap();
    let mse_fasq = recon_mse(&fasq, &queries).unwrap();
    println!("recon MSE          SQ8={:.6e}  SQ4={:.6e}  FASQ={:.6e}", mse_sq8, mse_sq4, mse_fasq);
    let ratio = mse_sq4 / mse_fasq;
    println!("FASQ vs SQ4 (same storage): {:.2}× lower MSE", ratio);

    // Encode base.
    let mut codes_sq8: Vec<Vec<u8>> = Vec::with_capacity(base.len());
    let mut codes_sq4: Vec<Vec<u8>> = Vec::with_capacity(base.len());
    let mut codes_fasq: Vec<Vec<u8>> = Vec::with_capacity(base.len());
    let t0 = Instant::now();
    for v in &base {
        let mut c = Vec::with_capacity(sq8.bytes_per_vector());
        sq8.encode(v, &mut c).unwrap();
        codes_sq8.push(c);
    }
    let enc_sq8 = t0.elapsed();
    let t0 = Instant::now();
    for v in &base {
        let mut c = Vec::with_capacity(sq4.bytes_per_vector());
        sq4.encode(v, &mut c).unwrap();
        codes_sq4.push(c);
    }
    let enc_sq4 = t0.elapsed();
    let t0 = Instant::now();
    for v in &base {
        let mut c = Vec::with_capacity(fasq.bytes_per_vector());
        fasq.encode(v, &mut c).unwrap();
        codes_fasq.push(c);
    }
    let enc_fasq = t0.elapsed();
    println!("encode {} vecs     SQ8={:.2?}  SQ4={:.2?}  FASQ={:.2?}", base.len(), enc_sq8, enc_sq4, enc_fasq);

    // Recall@10.
    let k = 10;
    let mut rec_sq8 = 0.0f32;
    let mut rec_sq4 = 0.0f32;
    let mut rec_fasq = 0.0f32;
    let t0 = Instant::now();
    for q in &queries {
        let gt = brute_force_knn(&base, q, k);
        rec_sq8 += recall_at_k(&gt, &quantized_knn(&sq8, &codes_sq8, &base, q, k));
        rec_sq4 += recall_at_k(&gt, &quantized_knn(&sq4, &codes_sq4, &base, q, k));
        rec_fasq += recall_at_k(&gt, &quantized_knn(&fasq, &codes_fasq, &base, q, k));
    }
    let dur_recall = t0.elapsed();
    let m = queries.len() as f32;
    println!("recall@{k}          SQ8={:.4}  SQ4={:.4}  FASQ={:.4}   (query eval {:.2?})",
        rec_sq8 / m, rec_sq4 / m, rec_fasq / m, dur_recall);
}

fn main() {
    println!("ruvector-fasq :: end-to-end demo");
    // Two workloads:
    //  A) Anisotropic (favorable for FASQ) — decaying per-dim variance.
    //  B) Isotropic (uniform variance) — FASQ should degrade gracefully to SQ4-ish.
    run(4_000, 4_000, 200, 64, 42, "anisotropic-decay", gen_anisotropic);
    run(4_000, 4_000, 200, 64, 43, "isotropic-unit-normal", gen_isotropic);
}
