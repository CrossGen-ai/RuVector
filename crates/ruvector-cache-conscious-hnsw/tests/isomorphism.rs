//! Reordering must be a pure permutation → same search outputs.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_cache_conscious_hnsw::{
    build_graph, reorder_with, search,
    reorder::{Bfs, ReverseCuthillMcKee},
};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    (0..n * dim).map(|_| normal.sample(&mut rng)).collect()
}

#[test]
fn bfs_and_rcm_preserve_search_results() {
    let n = 800;
    let dim = 32;
    let vectors = synth(n, dim, 7);
    let g = build_graph(vectors.clone(), dim, 16, 64, 1234);
    let g_bfs = reorder_with(&g, &Bfs);
    let g_rcm = reorder_with(&g, &ReverseCuthillMcKee);

    // random queries — each variant should return the SAME set of vector ids
    // (because reordering only renames ids; but distances are on stored vecs).
    let mut rng = StdRng::seed_from_u64(9999);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let k = 10;
    let ef = 32;

    for _ in 0..20 {
        let q: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let base = search(&g, &q, k, ef);
        let b_bfs = search(&g_bfs, &q, k, ef);
        let b_rcm = search(&g_rcm, &q, k, ef);

        // Distances (up to reordering-induced tie-break) should form the SAME multiset.
        let mut ds_a: Vec<f32> = base.iter().map(|x| x.0).collect();
        let mut ds_b: Vec<f32> = b_bfs.iter().map(|x| x.0).collect();
        let mut ds_c: Vec<f32> = b_rcm.iter().map(|x| x.0).collect();
        ds_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ds_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ds_c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(ds_a, ds_b, "BFS reorder changed distances");
        assert_eq!(ds_a, ds_c, "RCM reorder changed distances");
    }
}

#[test]
fn edge_span_shrinks_after_reorder() {
    let n = 2000;
    let dim = 32;
    let vectors = synth(n, dim, 7);
    let g = build_graph(vectors, dim, 24, 96, 1234);
    let base = ruvector_cache_conscious_hnsw::mean_edge_span(&g);
    let g_bfs = reorder_with(&g, &Bfs);
    let g_rcm = reorder_with(&g, &ReverseCuthillMcKee);
    let s_bfs = ruvector_cache_conscious_hnsw::mean_edge_span(&g_bfs);
    let s_rcm = ruvector_cache_conscious_hnsw::mean_edge_span(&g_rcm);
    // On random-gaussian data the graph fans out fully within 2 BFS hops, so
    // the span reduction is modest (real clustered data benefits far more).
    // Require strict reduction only.
    assert!(s_bfs < base, "BFS should reduce mean edge span (base={}, bfs={})", base, s_bfs);
    assert!(s_rcm < base, "RCM should reduce mean edge span (base={}, rcm={})", base, s_rcm);
}
