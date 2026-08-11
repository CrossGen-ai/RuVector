use crate::{build_index, data, search, AlphaPrune, Naive, RngPrune};

fn brute_top1(vectors: &[Vec<f32>], q: &[f32]) -> usize {
    let mut best = (0usize, f32::INFINITY);
    for (i, v) in vectors.iter().enumerate() {
        let d = crate::dist2(v, q);
        if d < best.1 { best = (i, d); }
    }
    best.0
}

#[test]
fn naive_index_builds_and_searches() {
    let vs = data::mixture(200, 16, 8, 42);
    let idx = build_index(&vs, &Naive, 8, 24);
    assert_eq!(idx.adj.len(), 200);
    assert!(idx.avg_degree > 0.0);
    // Query with a vector in the SAME cluster as the entry so a top-M
    // clustered graph can reach it. (Naive top-M is known to be cluster-local.)
    let q = &vs[8]; // same cluster as vs[0] under the mixture layout (i % c).
    let (res, _) = search(&idx, &vs, 0, q, 5, 64);
    assert_eq!(res.len(), 5);
    assert_eq!(res[0].0, 8);
}

#[test]
fn rng_prune_produces_diverse_edges() {
    let vs = data::mixture(300, 16, 8, 7);
    let idx = build_index(&vs, &RngPrune, 8, 32);
    // RNG pruning must not exceed M out-edges per node in the *forward* select;
    // back-links can push some nodes to exactly M after re-prune.
    for adj in &idx.adj {
        assert!(adj.len() <= 8, "degree {} exceeds M", adj.len());
    }
}

#[test]
fn alpha_1_matches_rng_shape() {
    let vs = data::mixture(150, 8, 5, 3);
    let a = build_index(&vs, &RngPrune, 6, 20);
    let b = build_index(&vs, &AlphaPrune::new(1.0), 6, 20);
    // Same pruning rule → same avg degree within tiny float ties.
    assert!((a.avg_degree - b.avg_degree).abs() < 0.5);
}

#[test]
fn recall_improves_with_ef() {
    // RNG pruning creates cross-cluster edges (diverse neighborhood), so a
    // wider ef should demonstrably lift recall over a narrow ef.
    let vs = data::mixture(500, 24, 10, 11);
    let idx = build_index(&vs, &RngPrune, 12, 40);
    let queries: Vec<&Vec<f32>> = (0..50).map(|i| &vs[i * 7 % 500]).collect();
    let mut hit_low = 0;
    let mut hit_high = 0;
    for q in &queries {
        let gt = brute_top1(&vs, q);
        let (r1, _) = search(&idx, &vs, 0, q, 1, 4);
        let (r2, _) = search(&idx, &vs, 0, q, 1, 128);
        if r1[0].0 == gt { hit_low += 1; }
        if r2[0].0 == gt { hit_high += 1; }
    }
    assert!(hit_high >= hit_low, "expected wider ef to help: low={hit_low} high={hit_high}");
    // Sanity floor: with RNG diversification + ef=128, majority of the 50
    // self-queries should be found.
    assert!(hit_high >= 25, "recall too low: {hit_high}/50");
}

#[test]
fn alpha_gt_1_keeps_more_or_equal_edges_than_rng() {
    let vs = data::mixture(400, 16, 8, 21);
    let rng_idx = build_index(&vs, &RngPrune, 12, 40);
    let a_idx   = build_index(&vs, &AlphaPrune::new(1.3), 12, 40);
    // Alpha > 1 relaxes the domination test, so it should keep >= as many edges.
    assert!(a_idx.avg_degree >= rng_idx.avg_degree - 0.05,
        "alpha={} rng={}", a_idx.avg_degree, rng_idx.avg_degree);
}
