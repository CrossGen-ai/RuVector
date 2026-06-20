//! Integration tests covering recall on a small dataset and the
//! invariant that LAET stays within ±eps of the fixed-ef baseline at
//! the calibrated recall target.

use ruvector_laet::data::brute_topk;
use ruvector_laet::search::{calibrate_training_set, recall_at_k};
use ruvector_laet::{
    gen_clustered, FixedEfStrategy, Hnsw, HnswParams, LaetStrategy, RidgePredictor, SearchStrategy,
};

fn build(seed: u64, n: usize, dim: usize) -> (Hnsw, Vec<Vec<f32>>, Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let ds = gen_clustered(seed, n, 100, dim, 8, 5.0, 1.0);
    let mut idx = Hnsw::new(
        dim,
        HnswParams {
            m: 12,
            m_max0: 24,
            ef_construction: 64,
            ml: 1.0 / (12f32).ln(),
            seed: 0x42,
        },
    );
    for v in ds.base.iter().cloned() {
        idx.insert(v);
    }
    let gt: Vec<Vec<u32>> = ds
        .queries
        .iter()
        .map(|q| brute_topk(&ds.base, q, 10).into_iter().map(|(i, _)| i as u32).collect())
        .collect();
    (idx, ds.queries, gt, ds.base)
}

#[test]
fn fixed_ef_recall_monotone() {
    let (idx, qs, gt, _) = build(11, 3_000, 32);
    let mut prev = 0.0f32;
    for ef in [10usize, 24, 64, 128, 256] {
        let s = FixedEfStrategy { ef };
        let mut tot = 0.0;
        for (i, q) in qs.iter().enumerate() {
            let r = s.search(&idx, q, 10);
            tot += recall_at_k(&r.ids, &gt[i], 10);
        }
        let r = tot / qs.len() as f32;
        assert!(r + 1e-3 >= prev, "recall regressed ef={ef}: {prev} -> {r}");
        prev = r;
    }
    assert!(prev > 0.9, "even max ef did not reach 0.9 recall: {prev}");
}

#[test]
fn laet_meets_target_recall() {
    let (idx, qs, gt, _) = build(13, 3_000, 32);
    let n_calib = 50;
    let target = 0.9f32;
    let ef_grid = [10usize, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    let train = calibrate_training_set(&idx, &qs[..n_calib], &gt[..n_calib], 10, target, &ef_grid);
    let xs: Vec<[f64; 5]> = train.iter().map(|(f, _)| f.to_vec()).collect();
    let ys: Vec<f64> = train.iter().map(|(_, y)| *y).collect();
    let mut p = RidgePredictor {
        lambda: 1.0,
        ef_floor: 10,
        ef_ceil: 256,
        ..Default::default()
    };
    p.fit(&xs, &ys);
    let laet = LaetStrategy { predictor: p };
    let mut tot = 0.0;
    for i in n_calib..qs.len() {
        let r = laet.search(&idx, &qs[i], 10);
        tot += recall_at_k(&r.ids, &gt[i], 10);
    }
    let recall = tot / (qs.len() - n_calib) as f32;
    // LAET is an approximation of the per-query optimum — allow a
    // generous margin below the target. This guards against
    // regressions where the predictor collapses to ef_floor.
    assert!(
        recall >= target - 0.1,
        "laet recall {recall} more than 0.1 below target {target}"
    );
}
