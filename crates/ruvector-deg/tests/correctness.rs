use ruvector_deg::{AnnIndex, BruteForce, Deg, KnnGraph, recall_at_k};
use rand::prelude::*;
use rand::rngs::StdRng;

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect()
}

#[test]
fn brute_force_self_match() {
    let dim = 16;
    let data = synth(50, dim, 7);
    let mut bf = BruteForce::new(dim);
    for v in &data { bf.insert(v.clone()); }
    for (i, v) in data.iter().enumerate() {
        let r = bf.search(v, 1);
        assert_eq!(r[0].0, i, "brute force must find the point itself first");
        assert!(r[0].1 < 1e-6);
    }
}

#[test]
fn deg_high_recall_on_small_cloud() {
    let dim = 16;
    let n = 600;
    let k = 10;
    let data = synth(n, dim, 11);
    let queries = synth(40, dim, 99);

    let mut bf = BruteForce::new(dim);
    for v in &data { bf.insert(v.clone()); }
    let truth: Vec<_> = queries.iter().map(|q| bf.search(q, k)).collect();

    let mut deg = Deg::new(dim, 24, 64, 64);
    for v in &data { deg.insert(v.clone()); }

    let mut sum = 0.0f32;
    for (qi, q) in queries.iter().enumerate() {
        sum += recall_at_k(&deg.search(q, k), &truth[qi], k);
    }
    let mean = sum / queries.len() as f32;
    assert!(mean >= 0.80, "DEG recall@10 = {mean} below 0.80 threshold");
}

#[test]
fn knn_graph_search_returns_k() {
    let dim = 8;
    let n = 100;
    let data = synth(n, dim, 5);
    let mut kg = KnnGraph::new(dim, 16, 32);
    for v in &data { kg.insert(v.clone()); }
    kg.build();
    let r = kg.search(&data[0], 5);
    assert_eq!(r.len(), 5);
    assert_eq!(r[0].0, 0);
}

#[test]
fn deg_degree_capped() {
    let dim = 8;
    let n = 300;
    let m = 12;
    let data = synth(n, dim, 3);
    let mut deg = Deg::new(dim, m, 32, 32);
    for v in &data { deg.insert(v.clone()); }
    for adj in &deg.adj {
        assert!(adj.len() <= m, "node exceeded max_degree: {} > {}", adj.len(), m);
    }
}
