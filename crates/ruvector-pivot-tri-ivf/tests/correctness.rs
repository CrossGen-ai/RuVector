use ruvector_pivot_tri_ivf::{gen_dataset, IvfIndex};

#[test]
fn pivot_prune_preserves_results_vs_baseline() {
    let data = gen_dataset(3000, 48, 123);
    let idx = IvfIndex::build(&data, 40, 6, 8, 123);
    let queries = gen_dataset(32, 48, 7);
    for q in &queries {
        let (base, _) = idx.search(q, 10, 10, false);
        let (prun, s) = idx.search(q, 10, 10, true);
        assert_eq!(base.len(), prun.len());
        for i in 0..base.len() {
            assert_eq!(base[i].id, prun[i].id, "mismatch at {i}");
            assert!((base[i].dist_sq - prun[i].dist_sq).abs() < 1e-5);
        }
        assert_eq!(s.candidates_seen, s.full_distances + s.pivot_prunes);
    }
}

#[test]
fn more_pivots_prune_at_least_as_much() {
    let data = gen_dataset(4000, 64, 55);
    let queries = gen_dataset(20, 64, 56);
    let idx4 = IvfIndex::build(&data, 48, 4, 8, 55);
    let idx8 = IvfIndex::build(&data, 48, 8, 8, 55);
    let mut p4 = 0usize;
    let mut p8 = 0usize;
    for q in &queries {
        let (_, s4) = idx4.search(q, 10, 8, true);
        let (_, s8) = idx8.search(q, 10, 8, true);
        p4 += s4.pivot_prunes;
        p8 += s8.pivot_prunes;
    }
    // m=8 should prune more candidates than m=4 overall (tighter LB).
    assert!(p8 >= p4, "p4={p4} p8={p8}");
}
