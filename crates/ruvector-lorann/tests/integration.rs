//! Integration tests covering all three backends end-to-end with real data.

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

#[test]
fn lorann_beats_random_recall() {
    let n = 1_000;
    let d = 32;
    let n_queries = 30;
    let k = 10;
    let x = synth(n, d, 1);
    let queries = synth(n_queries, d, 2);

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

    let cfg = LoRannConfig {
        ivf: IvfConfig { n_clusters: 16, nprobe: 4, kmeans_iters: 25, seed: 7 },
        rank: 16,
        subspace_iters: 20,
        rerank_per_cluster: 24,
    };
    let mut lorann = LoRannIndex::new(cfg);
    lorann.train(x.clone(), n, d);
    let pred: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            lorann
                .search(&queries[i * d..(i + 1) * d], k)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();
    let recall = recall_at_k(&truth, &pred, k);
    let random_baseline = k as f32 / n as f32;
    assert!(
        recall > 10.0 * random_baseline,
        "lorann recall@{k}={recall} should beat 10x random baseline {random_baseline}"
    );
}

#[test]
fn brute_force_top1_is_self() {
    let n = 64;
    let d = 16;
    let x = synth(n, d, 3);
    let mut brute = BruteForceIndex::default();
    brute.train(x.clone(), n, d);
    for i in 0..10 {
        let q = &x[i * d..(i + 1) * d];
        let top = brute.search(q, 1);
        assert_eq!(top[0].id, i as u32);
    }
}

#[test]
fn ivf_recovers_majority_of_ground_truth() {
    let n = 500;
    let d = 24;
    let k = 10;
    let n_queries = 20;
    let x = synth(n, d, 5);
    let queries = synth(n_queries, d, 6);
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
    let mut ivf = IvfIndex::new(IvfConfig { n_clusters: 8, nprobe: 4, kmeans_iters: 20, seed: 9 });
    ivf.train(x.clone(), n, d);
    let pred: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            ivf.search(&queries[i * d..(i + 1) * d], k)
                .into_iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();
    let recall = recall_at_k(&truth, &pred, k);
    assert!(recall > 0.4, "ivf recall@{k} too low: {recall}");
}
