use ruvector_leanvec_shortlist::{
    gen_clustered, recall_at_k, BruteForce, LeanVecShortlist, Projection, Searcher, XorShift64,
};

#[test]
fn projection_is_orthonormal() {
    let p = Projection::random_gaussian(64, 8, 42);
    // Rows should each have L2 norm ≈ 1.
    for r in 0..p.d_out {
        let row = &p.rows[r * p.d_in..(r + 1) * p.d_in];
        let n2: f32 = row.iter().map(|v| v * v).sum::<f32>();
        assert!((n2 - 1.0).abs() < 1e-3, "row {} norm^2 = {}", r, n2);
    }
}

#[test]
fn pca_rows_are_orthonormal() {
    let ds = gen_clustered(1000, 64, 8, 10, 7);
    let p = Projection::learn_pca(&ds, 8, 4, 500, 7);
    // Orthogonality: Q Q^T ≈ I (8x8).
    for i in 0..p.d_out {
        for j in 0..p.d_out {
            let ri = &p.rows[i * p.d_in..(i + 1) * p.d_in];
            let rj = &p.rows[j * p.d_in..(j + 1) * p.d_in];
            let dot: f32 = ri.iter().zip(rj.iter()).map(|(a, b)| a * b).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-2,
                "Q Q^T[{}][{}] = {}, expected {}",
                i,
                j,
                dot,
                expected
            );
        }
    }
}

#[test]
fn leanvec_shortlist_matches_brute_force_at_recall() {
    let n = 2_000;
    let d = 64;
    let k = 10;
    let ds = gen_clustered(n, d, 16, 8, 3);
    let bf = BruteForce { ds: &ds };

    let proj = Projection::learn_pca(&ds, 16, 5, 800, 3);
    let lv = LeanVecShortlist::build(&ds, proj, 100, "lv");

    let mut rng = XorShift64::new(99);
    let mut total_recall = 0.0f32;
    let n_q = 30;
    for _ in 0..n_q {
        let src = (rng.next_u64() as usize) % n;
        let row = ds.row(src);
        let q: Vec<f32> = row.iter().map(|v| v + 0.25 * rng.normal()).collect();
        let truth = bf.search(&q, k);
        let pred = lv.search(&q, k);
        total_recall += recall_at_k(&pred, &truth, k);
    }
    total_recall /= n_q as f32;
    // With clustered data and L=100 >> k=10, recall should be well above 0.7.
    assert!(total_recall > 0.7, "avg recall {} too low", total_recall);
}
