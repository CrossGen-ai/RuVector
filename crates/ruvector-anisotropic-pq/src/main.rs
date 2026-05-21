//! anisotropic-pq-demo
//!
//! Generates a synthetic anisotropic Gaussian dataset (mixture of clusters with
//! one elongated direction per cluster), trains all three quantizers, and
//! prints code size, build time, and top-10 cosine recall vs exact.
//!
//! This is the canonical reproducible run. Numbers printed here are the
//! ones quoted in the research doc and ADR.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Instant;

use ruvector_anisotropic_pq::metrics::{dot, normalize_in_place};
use ruvector_anisotropic_pq::{AnisotropicPq, Opq, Pq, Quantizer};

fn gen_dataset(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // unit-norm cluster centers (so the dataset lives on / near S^{d-1},
    // matching how dense retrieval embeddings are typically used)
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng) as f32).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect();
    let long_axes: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng) as f32).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect();

    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.gen_range(0..n_clusters);
        let center = &centers[c];
        let axis = &long_axes[c];
        let mut v = vec![0f32; d];
        // small isotropic jitter
        for i in 0..d {
            v[i] = center[i] + (normal.sample(&mut rng) as f32) * 0.15;
        }
        // anisotropic elongation along the cluster's long axis
        let along = normal.sample(&mut rng) as f32 * 0.6;
        for i in 0..d {
            v[i] += along * axis[i];
        }
        normalize_in_place(&mut v);
        out.push(v);
    }
    out
}

fn exact_top_k_ip(db: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    // Vectors are unit-normalized → inner product == cosine similarity.
    let mut scored: Vec<(usize, f32)> =
        db.iter().enumerate().map(|(i, v)| (i, dot(q, v))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn pq_top_k<Q: Quantizer>(
    quantizer: &Q,
    codes: &[Vec<u8>],
    query: &[f32],
    k: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, quantizer.asymmetric_score(query, c)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall_at_k(predicted: &[usize], truth: &[usize]) -> f32 {
    let mut hit = 0;
    for p in predicted {
        if truth.contains(p) {
            hit += 1;
        }
    }
    hit as f32 / truth.len() as f32
}

fn main() {
    let n_train = 4_000;
    let n_db = 8_000;
    let n_queries = 200;
    let d = 64;
    let m = 8;
    let k = 256;
    let kmeans_iters = 25;
    let opq_sweeps = 3;
    let apq_sweeps = 3;
    let eta = 4.0;

    println!("== anisotropic-pq demo ==");
    println!(
        "dataset: n_train={n_train}, n_db={n_db}, n_queries={n_queries}, d={d}\n\
         quantizer:   m={m}, k={k}, code_bytes={m}, eta={eta}\n"
    );

    let train = gen_dataset(n_train, d, 16, 7);
    let db = gen_dataset(n_db, d, 16, 17);
    let queries = gen_dataset(n_queries, d, 16, 19);

    // train all three
    println!("training PQ ...");
    let t = Instant::now();
    let pq = Pq::train(&train, m, k, kmeans_iters, 1).unwrap();
    let pq_train_ms = t.elapsed().as_millis();
    println!("  done in {pq_train_ms} ms");

    println!("training OPQ ...");
    let t = Instant::now();
    let opq = Opq::train(&train, m, k, kmeans_iters, opq_sweeps, 2).unwrap();
    let opq_train_ms = t.elapsed().as_millis();
    println!("  done in {opq_train_ms} ms");

    println!("training Anisotropic-PQ (eta={eta}) ...");
    let t = Instant::now();
    let apq = AnisotropicPq::train(&train, m, k, eta, kmeans_iters, apq_sweeps, 3).unwrap();
    let apq_train_ms = t.elapsed().as_millis();
    println!("  done in {apq_train_ms} ms");

    // encode db
    let t = Instant::now();
    let pq_codes: Vec<Vec<u8>> = db.iter().map(|v| pq.encode(v)).collect();
    let pq_enc_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let opq_codes: Vec<Vec<u8>> = db.iter().map(|v| opq.encode(v)).collect();
    let opq_enc_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let apq_codes: Vec<Vec<u8>> = db.iter().map(|v| apq.encode(v)).collect();
    let apq_enc_ms = t.elapsed().as_millis();

    println!("\nencode {n_db} db vectors:");
    println!("  PQ      {pq_enc_ms} ms");
    println!("  OPQ     {opq_enc_ms} ms");
    println!("  APQ     {apq_enc_ms} ms");

    // recall vs exact cosine top-10
    let k_recall = 10;
    let mut pq_recall = 0f32;
    let mut opq_recall = 0f32;
    let mut apq_recall = 0f32;
    let t = Instant::now();
    for q in &queries {
        let truth = exact_top_k_ip(&db, q, k_recall);
        pq_recall += recall_at_k(&pq_top_k(&pq, &pq_codes, q, k_recall), &truth);
        opq_recall += recall_at_k(&pq_top_k(&opq, &opq_codes, q, k_recall), &truth);
        apq_recall += recall_at_k(&pq_top_k(&apq, &apq_codes, q, k_recall), &truth);
    }
    let search_ms = t.elapsed().as_millis();
    pq_recall /= n_queries as f32;
    opq_recall /= n_queries as f32;
    apq_recall /= n_queries as f32;

    println!("\nrecall @ {k_recall} (over {n_queries} queries, exact IP ground truth, unit-norm data):");
    println!("  PQ      {:.4}", pq_recall);
    println!("  OPQ     {:.4}", opq_recall);
    println!("  APQ     {:.4}   (eta={eta})", apq_recall);
    println!("\nfull recall sweep took {search_ms} ms");

    let raw_bytes = n_db * d * 4;
    let code_bytes = n_db * m;
    println!(
        "\ncompression: raw f32 = {raw_bytes} B,  PQ codes = {code_bytes} B  ({:.1}x smaller)",
        raw_bytes as f32 / code_bytes as f32
    );
}
