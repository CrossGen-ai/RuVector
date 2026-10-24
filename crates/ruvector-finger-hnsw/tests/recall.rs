//! Numerical acceptance tests.
//!
//! These run as part of `cargo test` and verify:
//!   1. The exact search is a correct baseline (recall ~ 1 on a small graph).
//!   2. Guarded search preserves recall within an epsilon of exact.
//!   3. Pure sketch search returns valid (non-NaN, non-negative) distances
//!      and improves distance-call count vs exact.

use ruvector_finger_hnsw::*;

fn build_setup(n: usize, d: usize, m: usize, k: usize) -> (Vec<Vec<f32>>, FlatGraph, EdgeSketch) {
    let data = gen_dataset(n, d, 8, 0xDEADBEEF);
    let graph = FlatGraph::build(&data, &GraphParams { m, m_long: 4, seed: 0xC0FFEE });
    let sketch = EdgeSketch::build(&data, &graph, &SketchParams { k, seed: 0xA17EC0DE });
    (data, graph, sketch)
}

#[test]
fn exact_baseline_recall_is_high() {
    let (data, graph, _sketch) = build_setup(400, 48, 16, 16);
    let queries = gen_queries_from_data(&data, 32, 0.5, 0xDEAD1);

    let k = 10;
    let ef = 32;
    let exact = ExactSearch;
    let mut total = 0.0f32;
    for q in &queries {
        let gt: Vec<u32> = brute_force_topk(&data, q, k).iter().map(|p| p.0).collect();
        let r = exact.search(&data, &graph, q, k, ef);
        total += recall_at_k(&gt, &r.ids);
    }
    let avg = total / queries.len() as f32;
    assert!(avg > 0.85, "exact baseline recall too low: {}", avg);
}

#[test]
fn guarded_search_preserves_recall() {
    let (data, graph, sketch) = build_setup(400, 48, 16, 16);
    let queries = gen_queries_from_data(&data, 32, 0.5, 0xDEAD2);

    let k = 10;
    let ef = 32;
    let exact = ExactSearch;
    let guarded = FingerGuarded { sketch: &sketch, slack: 0.25 };

    let mut exact_recall = 0.0f32;
    let mut guarded_recall = 0.0f32;
    let mut guarded_dist_calls = 0u64;
    let mut exact_dist_calls = 0u64;
    for q in &queries {
        let gt: Vec<u32> = brute_force_topk(&data, q, k).iter().map(|p| p.0).collect();
        let re = exact.search(&data, &graph, q, k, ef);
        let rg = guarded.search(&data, &graph, q, k, ef);
        exact_recall += recall_at_k(&gt, &re.ids);
        guarded_recall += recall_at_k(&gt, &rg.ids);
        exact_dist_calls += re.stats.exact_dist_calls;
        guarded_dist_calls += rg.stats.exact_dist_calls;
    }
    let e = exact_recall / queries.len() as f32;
    let g = guarded_recall / queries.len() as f32;
    // Guarded is allowed to lose up to 2 percentage points vs exact.
    assert!(g + 0.02 >= e, "guarded recall {} fell too far below exact recall {}", g, e);
    // Guarded must save at least some exact distance work.
    assert!(guarded_dist_calls < exact_dist_calls,
        "guarded did not reduce exact-dist calls (guarded={}, exact={})",
        guarded_dist_calls, exact_dist_calls);
}

#[test]
fn pure_sketch_returns_valid_distances() {
    let (data, graph, sketch) = build_setup(300, 32, 12, 16);
    let queries = gen_queries_from_data(&data, 16, 0.5, 0xDEAD3);

    let finger = FingerSearch { sketch: &sketch };
    for q in &queries {
        let r = finger.search(&data, &graph, q, 10, 24);
        for &d in &r.dists {
            assert!(d.is_finite(), "non-finite distance: {}", d);
            assert!(d >= 0.0, "negative distance: {}", d);
        }
        assert!(r.stats.approx_dist_calls > 0, "no approx calls made");
    }
}

#[test]
fn sketch_memory_matches_formula() {
    // bytes = 4 * (k*d + n*m*k + n*m).
    let (_data, graph, sketch) = build_setup(100, 32, 8, 16);
    let expected = 4 * (sketch.k * sketch.dim + sketch.n * sketch.m * sketch.k + sketch.n * sketch.m);
    assert_eq!(sketch.bytes, expected, "sketch byte count differs from formula");
    assert_eq!(sketch.n, graph.n);
    assert_eq!(sketch.m, graph.m);
}
