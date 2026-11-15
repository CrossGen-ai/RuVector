//! Correctness tests — real data, no mocks.

use rvq_early_exit::{
    gen_clustered, recall_at_k, scan_early_exit, scan_exact, scan_full, Rng, Rvq,
};

#[test]
fn rvq_full_matches_reconstruction() {
    // On a small set, full-depth ADC's ranking should align closely with
    // re-ranking by exact distance to the reconstructed codes.
    let mut rng = Rng::new(42);
    let data = gen_clustered(1000, 32, 16, &mut rng);
    let rvq = Rvq::train_and_encode(&data, 4, 32, 10, &mut rng);
    let pn = rvq.build_partial_norms();
    let q = &data[7]; // query is in-set; should rank itself high
    let hits = scan_full(&rvq, q, &pn, 10);
    assert!(hits.iter().any(|h| h.idx == 7), "self must appear in top-10");
    // Top-1 must have a sensible non-negative squared-distance estimate.
    assert!(hits[0].d >= -1e-3, "partial dist shouldn't be hugely negative: {}", hits[0].d);
}

#[test]
fn early_exit_is_shortlist_consistent_with_full() {
    // Early exit returns a subset-of-shortlist ranking. On easy data, its
    // top-K should overlap heavily with full-depth top-K.
    let mut rng = Rng::new(123);
    let data = gen_clustered(5000, 64, 32, &mut rng);
    let rvq = Rvq::train_and_encode(&data, 6, 128, 10, &mut rng);
    let pn = rvq.build_partial_norms();
    let gt_queries: Vec<_> = (0..20)
        .map(|_| {
            let i = (rng.next_u64() as usize) % data.len();
            data[i].clone()
        })
        .collect();
    let mut overlap_sum = 0.0f32;
    for q in &gt_queries {
        let full = scan_full(&rvq, q, &pn, 10);
        let (ee, _) = scan_early_exit(&rvq, q, &pn, 10, 8, 2);
        overlap_sum += recall_at_k(&full, &ee);
    }
    let avg = overlap_sum / gt_queries.len() as f32;
    assert!(avg >= 0.7, "early-exit vs full overlap should be ≥0.7, got {}", avg);
}

#[test]
fn recall_against_exact_is_meaningful() {
    let mut rng = Rng::new(7);
    let data = gen_clustered(3000, 64, 24, &mut rng);
    let rvq = Rvq::train_and_encode(&data, 6, 128, 12, &mut rng);
    let pn = rvq.build_partial_norms();
    let queries: Vec<_> = (0..30)
        .map(|_| {
            let i = (rng.next_u64() as usize) % data.len();
            let mut v = data[i].clone();
            for t in 0..v.len() {
                v[t] += rng.gaussian() * 0.2;
            }
            v
        })
        .collect();
    let gts: Vec<_> = queries.iter().map(|q| scan_exact(&data, q, 10)).collect();
    let mut r_full = 0.0;
    let mut r_ee = 0.0;
    for (qi, q) in queries.iter().enumerate() {
        let full = scan_full(&rvq, q, &pn, 10);
        let (ee, _) = scan_early_exit(&rvq, q, &pn, 10, 8, 2);
        r_full += recall_at_k(&gts[qi], &full);
        r_ee += recall_at_k(&gts[qi], &ee);
    }
    r_full /= queries.len() as f32;
    r_ee /= queries.len() as f32;
    // Loose but meaningful threshold: on 3k pts / D=64 / M=6 / K=128 with noisy
    // in-set queries, ADC recall should materially beat random (random@10 ≈ 10/3000).
    assert!(r_full >= 0.3, "full-ADC recall should be ≥0.3, got {}", r_full);
    assert!(r_ee >= 0.3, "early-exit recall should be ≥0.3, got {}", r_ee);
}

#[test]
fn partial_norm_consistency() {
    // partial_norms[i][m] should equal ||reconstruct(i)||^2 exactly.
    let mut rng = Rng::new(9);
    let data = gen_clustered(200, 32, 8, &mut rng);
    let rvq = Rvq::train_and_encode(&data, 4, 16, 8, &mut rng);
    let pn = rvq.build_partial_norms();
    for i in 0..rvq.n {
        let rec = rvq.reconstruct(i);
        let nn: f32 = rec.iter().map(|x| x * x).sum();
        let stored = pn[i * (rvq.m + 1) + rvq.m];
        let diff = (nn - stored).abs();
        assert!(diff < 1e-2, "mismatch at {}: {} vs {}", i, nn, stored);
    }
}
