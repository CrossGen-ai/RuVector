//! `cargo run --release -p ruvector-lorann --example demo`
//!
//! Builds a synthetic dataset, trains all three backends, prints recall@10 of
//! LoRANN and IVF vs the brute-force ground truth, and a single example query.

use ruvector_lorann::{
    BruteForceIndex, InnerProductIndex, IvfConfig, IvfIndex, Lcg, LoRannConfig, LoRannIndex,
};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut x = vec![0.0_f32; n * d];
    for k in 0..n * d {
        x[k] = rng.next_f32() - 0.5;
    }
    x
}

fn recall_at_k(truth: &[Vec<u32>], pred: &[Vec<u32>], k: usize) -> f32 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (t, p) in truth.iter().zip(pred.iter()) {
        let pset: std::collections::HashSet<u32> = p.iter().copied().take(k).collect();
        for &id in t.iter().take(k) {
            if pset.contains(&id) {
                hits += 1;
            }
            total += 1;
        }
    }
    hits as f32 / total.max(1) as f32
}

fn main() {
    let n = 4_000;
    let d = 64;
    let n_queries = 100;
    let k = 10;

    println!("ruvector-lorann demo  n={n}  d={d}  queries={n_queries}  k={k}");

    let x = synth(n, d, 1);
    let queries = synth(n_queries, d, 2);

    // Brute force ground truth.
    let mut brute = BruteForceIndex::default();
    brute.train(x.clone(), n, d);
    let truth: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            brute
                .search(&queries[i * d..(i + 1) * d], k)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();

    // IVF.
    let ivf_cfg = IvfConfig { n_clusters: 32, nprobe: 4, kmeans_iters: 25, seed: 11 };
    let mut ivf = IvfIndex::new(ivf_cfg);
    ivf.train(x.clone(), n, d);
    let ivf_pred: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            ivf.search(&queries[i * d..(i + 1) * d], k)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();

    // LoRANN.
    let lorann_cfg = LoRannConfig {
        ivf: ivf_cfg,
        rank: 16,
        subspace_iters: 25,
        rerank_per_cluster: 32,
    };
    let mut lorann = LoRannIndex::new(lorann_cfg);
    lorann.train(x.clone(), n, d);
    let lorann_pred: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            lorann
                .search(&queries[i * d..(i + 1) * d], k)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();

    println!(
        "recall@{k}  ivf={:.3}  lorann={:.3}",
        recall_at_k(&truth, &ivf_pred, k),
        recall_at_k(&truth, &lorann_pred, k)
    );

    // Show one example.
    let _q0 = &queries[..d];
    println!("\nExample query (q0) top-{k} ids:");
    println!("  brute : {:?}", &truth[0]);
    println!("  ivf   : {:?}", &ivf_pred[0]);
    println!("  lorann: {:?}", &lorann_pred[0]);
}
