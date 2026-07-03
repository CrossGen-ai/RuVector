use ruvector_ptolemaic::{
    gen_dataset, gen_queries,
    pivot::{select_farthest_first, select_random},
    KnnIndex, LinearScan, PtolemaicPivot, TrianglePivot,
};

/// All three backends MUST return identical ordered top-k for identical
/// inputs — pruning is a speed optimisation, not an approximation.
#[test]
fn triangle_and_ptolemaic_match_linear() {
    let dim = 16;
    let n = 800;
    let k = 20;
    let n_q = 25;
    let data = gen_dataset(n, dim, 6, 11);
    let queries = gen_queries(n_q, dim, 22);
    let pivots = select_farthest_first(&data, dim, 6, 33);

    let lin = LinearScan::new(data.clone(), dim).unwrap();
    let tri = TrianglePivot::build(data.clone(), dim, pivots.clone()).unwrap();
    let pto = PtolemaicPivot::build(data.clone(), dim, pivots).unwrap();

    for q in 0..n_q {
        let qv = &queries[q * dim..(q + 1) * dim];
        let a = lin.search(qv, k).unwrap().0;
        let b = tri.search(qv, k).unwrap().0;
        let c = pto.search(qv, k).unwrap().0;
        let ids_a: Vec<u32> = a.iter().map(|n| n.id).collect();
        let ids_b: Vec<u32> = b.iter().map(|n| n.id).collect();
        let ids_c: Vec<u32> = c.iter().map(|n| n.id).collect();
        assert_eq!(ids_a, ids_b, "triangle differs at q={q}");
        assert_eq!(ids_a, ids_c, "ptolemaic differs at q={q}");
    }
}

/// Ptolemaic pruning must not be *worse* than triangle pruning: for any query
/// the number of pruned candidates should be >= triangle's.
#[test]
fn ptolemaic_dominates_triangle() {
    let dim = 32;
    let n = 1200;
    let k = 10;
    let n_q = 30;
    let data = gen_dataset(n, dim, 8, 111);
    let queries = gen_queries(n_q, dim, 222);
    let pivots = select_farthest_first(&data, dim, 8, 333);

    let tri = TrianglePivot::build(data.clone(), dim, pivots.clone()).unwrap();
    let pto = PtolemaicPivot::build(data.clone(), dim, pivots).unwrap();

    for q in 0..n_q {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, st) = tri.search(qv, k).unwrap();
        let (_, sp) = pto.search(qv, k).unwrap();
        assert!(
            sp.pruned >= st.pruned,
            "Ptolemaic pruned fewer than Triangle at q={q}: pto={} tri={}",
            sp.pruned,
            st.pruned
        );
    }
}

#[test]
fn random_pivots_still_correct() {
    let dim = 8;
    let n = 400;
    let k = 5;
    let n_q = 10;
    let data = gen_dataset(n, dim, 5, 1);
    let queries = gen_queries(n_q, dim, 2);
    let pivots = select_random(&data, dim, 4, 3);

    let lin = LinearScan::new(data.clone(), dim).unwrap();
    let pto = PtolemaicPivot::build(data.clone(), dim, pivots).unwrap();

    for q in 0..n_q {
        let qv = &queries[q * dim..(q + 1) * dim];
        let a: Vec<u32> = lin.search(qv, k).unwrap().0.iter().map(|n| n.id).collect();
        let b: Vec<u32> = pto.search(qv, k).unwrap().0.iter().map(|n| n.id).collect();
        assert_eq!(a, b);
    }
}
