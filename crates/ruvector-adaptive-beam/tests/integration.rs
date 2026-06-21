//! Integration tests: build a small HNSW and verify each terminator
//! produces valid top-k.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_adaptive_beam::{
    FixedEfTerminator, Hnsw, HnswParams, QuantileTerminator, RatioTerminator,
};

fn sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
    let mut v: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, sq(x, q)))
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall(data: &[Vec<f32>], idx: &Hnsw, queries: &[Vec<f32>], term: &mut dyn ruvector_adaptive_beam::BeamTerminator, k: usize) -> f64 {
    let mut hits = 0usize;
    let mut tot = 0usize;
    for q in queries {
        let truth = brute_topk(data, q, k);
        let (res, _) = idx.search(q, k, term);
        let ids: std::collections::HashSet<u32> = res.iter().map(|n| n.id).collect();
        for t in truth {
            tot += 1;
            if ids.contains(&t) {
                hits += 1;
            }
        }
    }
    hits as f64 / tot as f64
}

#[test]
fn all_terminators_meet_recall_floor() {
    let dim = 32;
    let n = 2_000;
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let data: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();
    let queries: Vec<Vec<f32>> = (0..50).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect();

    let mut idx = Hnsw::new(dim, HnswParams::default());
    for v in &data {
        idx.insert(v);
    }
    let k = 10;
    let r_fixed = recall(&data, &idx, &queries, &mut FixedEfTerminator::new(200), k);
    let r_ratio = recall(&data, &idx, &queries, &mut RatioTerminator::new(16, 1.10, 256), k);
    let r_quant = recall(&data, &idx, &queries, &mut QuantileTerminator::new(16, 0.75, 256), k);
    assert!(r_fixed > 0.80, "fixed recall too low: {r_fixed}");
    assert!(r_ratio > 0.70, "ratio recall too low: {r_ratio}");
    assert!(r_quant > 0.70, "quantile recall too low: {r_quant}");
}
