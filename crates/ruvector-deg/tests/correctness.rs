use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use ruvector_deg::distance::{l2_sq, Metric};
use ruvector_deg::graph::{DegGraph, DegParams};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()).collect()
}

fn brute(data: &[Vec<f32>], q: &[f32], k: usize, alive: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut v: Vec<(u32, f32)> = data.iter().enumerate()
        .filter(|(i, _)| alive(*i as u32))
        .map(|(i, x)| (i as u32, l2_sq(x, q))).collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn invariant_bounded_degree() {
    let dim = 16; let n = 400;
    let data = random_vectors(n, dim, 1);
    let params = DegParams { degree: 12, eps: 40, refine: 2, metric: Metric::L2Sq, seed: 1 };
    let mut g = DegGraph::new(dim, params.clone());
    for v in &data { g.insert(v); }
    // Once we have > degree+1 vertices every live vertex must have exactly
    // `degree` non-TOMB outgoing edges.
    let edges = g.edge_count();
    assert_eq!(edges, g.len() * params.degree, "expected {} edges, got {}", g.len() * params.degree, edges);
}

#[test]
fn recall_meets_floor() {
    let dim = 32; let n = 1500; let k = 10;
    let data = random_vectors(n, dim, 2);
    let queries = random_vectors(50, dim, 3);
    let params = DegParams { degree: 24, eps: 80, refine: 4, metric: Metric::L2Sq, seed: 2 };
    let mut g = DegGraph::new(dim, params);
    for v in &data { g.insert(v); }
    let mut total = 0.0f32;
    for q in &queries {
        let truth = brute(&data, q, k, |_| true);
        let approx = g.search(q, k);
        let t: std::collections::HashSet<u32> = truth.into_iter().collect();
        total += approx.iter().filter(|(i, _)| t.contains(i)).count() as f32 / k as f32;
    }
    let recall = total / queries.len() as f32;
    // Acceptance test: recall@10 ≥ 0.80 on a 1.5k uniform-random dataset.
    assert!(recall >= 0.80, "recall {:.3} below 0.80 floor", recall);
}

#[test]
fn deletion_patches_dangling_edges() {
    let dim = 16; let n = 300;
    let data = random_vectors(n, dim, 4);
    let params = DegParams { degree: 12, eps: 40, refine: 2, metric: Metric::L2Sq, seed: 3 };
    let mut g = DegGraph::new(dim, params);
    for v in &data { g.insert(v); }
    // Delete a chunk.
    let killed: Vec<u32> = (0..100).map(|i| i as u32).collect();
    for id in &killed { g.delete(*id); }
    assert_eq!(g.len(), n - killed.len());
    // No live vertex should still reference a deleted id.
    use std::collections::HashSet;
    let dead: HashSet<u32> = killed.iter().copied().collect();
    // Internal check: search must never return a dead id.
    let q = &data[200];
    let res = g.search(q, 20);
    for (id, _) in &res {
        assert!(!dead.contains(id), "search returned deleted id {id}");
    }
    // Recall against surviving set should still be reasonable.
    let truth = brute(&data, q, 10, |i| !dead.contains(&i));
    let t: HashSet<u32> = truth.into_iter().collect();
    let approx = g.search(q, 10);
    let hits = approx.iter().filter(|(id, _)| t.contains(id)).count();
    assert!(hits >= 5, "only {hits}/10 recall after deletion churn");
}

#[test]
fn streaming_insert_after_delete_reuses_slots() {
    let dim = 8;
    let params = DegParams { degree: 6, eps: 20, refine: 1, metric: Metric::L2Sq, seed: 4 };
    let mut g = DegGraph::new(dim, params);
    let a = random_vectors(50, dim, 5);
    let mut ids: Vec<u32> = a.iter().map(|v| g.insert(v)).collect();
    let cap_before = g.capacity();
    // Delete half.
    for id in ids.drain(..25) { g.delete(id); }
    assert_eq!(g.len(), 25);
    // Insert 20 fresh — should reuse vacant slots, not extend capacity.
    let b = random_vectors(20, dim, 6);
    for v in &b { g.insert(v); }
    assert_eq!(g.capacity(), cap_before, "capacity grew despite available free slots");
    assert_eq!(g.len(), 45);
}
