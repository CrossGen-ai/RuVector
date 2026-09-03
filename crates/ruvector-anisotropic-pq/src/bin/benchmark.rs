//! Real benchmark: train three quantizers on the same data, measure MIPS
//! Recall@10 and reconstruction MSE, print a Markdown table.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p ruvector-anisotropic-pq --bin benchmark
//! ```

use ruvector_anisotropic_pq::{
    recall_at_k, reconstruction_mse, AdcSearcher, AnisotropicPq, BaselinePq, DatasetConfig,
    GaussianMixture, PqParams, Quantizer,
};
use std::time::Instant;

fn main() {
    let ds_cfg = DatasetConfig::default();
    let params = PqParams::default();
    let top_k = 10usize;

    let banner = "─".repeat(76);
    println!("{banner}");
    println!("  ruvector-anisotropic-pq benchmark");
    println!("{banner}");
    println!(
        "  dataset : n_db={}  n_queries={}  dim={}  k_clusters={}  noise_std={}",
        ds_cfg.n_db, ds_cfg.n_queries, ds_cfg.dim, ds_cfg.k_clusters, ds_cfg.noise_std
    );
    println!(
        "  pq      : m={}  k={}  sub_dim={}  iters={}",
        params.m,
        params.k,
        params.dim / params.m,
        params.iters
    );
    println!("  metric  : MIPS Recall@{top_k}, reconstruction MSE");
    println!("{banner}");

    let t0 = Instant::now();
    let ds = GaussianMixture::generate(ds_cfg);
    let truth = ds.ground_truth_mips(top_k);
    println!(
        "  generated data + ground-truth top-{top_k} in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let quantizers: Vec<Box<dyn Quantizer>> = vec![
        Box::new(BaselinePq),
        Box::new(AnisotropicPq { eta: 4.0 }),
        Box::new(AnisotropicPq { eta: 16.0 }),
    ];

    println!(
        "\n| Quantizer | Train (s) | Encode (s) | Search (ms/q) | Recall@{top_k} | MSE |"
    );
    println!("|-----------|-----------|------------|---------------|-----------|-----|");

    for q in &quantizers {
        let t_train = Instant::now();
        let cb = q.train(&ds.db, params);
        let train_s = t_train.elapsed().as_secs_f64();

        let t_enc = Instant::now();
        let codes = cb.encode_batch(&ds.db);
        let enc_s = t_enc.elapsed().as_secs_f64();

        let searcher = AdcSearcher::new(&cb, &codes);
        let t_srch = Instant::now();
        let mut preds: Vec<Vec<u32>> = Vec::with_capacity(ds.queries.len());
        for qv in &ds.queries {
            preds.push(searcher.search(qv, top_k).into_iter().map(|h| h.id).collect());
        }
        let srch_us_per_q = t_srch.elapsed().as_secs_f64() * 1e3 / ds.queries.len() as f64;

        let recall = recall_at_k(&preds, &truth);
        let mse = reconstruction_mse(&cb, &ds.db, &codes);

        println!(
            "| {:<30} | {:>9.3} | {:>10.3} | {:>13.3} | {:>9.4} | {:>5.4} |",
            q.name(),
            train_s,
            enc_s,
            srch_us_per_q,
            recall,
            mse
        );
    }

    println!("{banner}");
    println!(
        "  total: {:.2}s   (single-thread, deterministic seed = {:#x})",
        t0.elapsed().as_secs_f64(),
        ds_cfg.seed
    );
    println!("{banner}");
}
