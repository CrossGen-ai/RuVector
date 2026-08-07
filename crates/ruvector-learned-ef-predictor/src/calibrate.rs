//! Calibration: for each calibration query, find the minimum `ef` (from a
//! discrete ladder) that hits the recall target vs brute-force top-k, then
//! use those (features, y) pairs to fit the linear predictor.

use crate::controller::LearnedLinearEf;
use crate::hnsw::Hnsw;

pub const EF_LADDER: &[usize] = &[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];

pub fn recall_at_k(pred: &[u32], truth: &[u32]) -> f32 {
    let mut hit = 0usize;
    for p in pred { if truth.contains(p) { hit += 1; } }
    hit as f32 / truth.len() as f32
}

pub fn calibrate(
    index: &Hnsw,
    calib_queries: &[Vec<f32>],
    k: usize,
    recall_target: f32,
) -> LearnedLinearEf {
    let mut xs: Vec<[f32; 6]> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();

    for q in calib_queries {
        let truth = index.brute_force(q, k);
        let probe = index.probe(q);
        let feats = LearnedLinearEf::features(&probe);

        // Find minimum ef in the ladder that reaches recall_target.
        let mut chosen: usize = *EF_LADDER.last().unwrap();
        for &ef in EF_LADDER {
            let (pred, _) = index.search(q, k, ef);
            if recall_at_k(&pred, &truth) >= recall_target {
                chosen = ef;
                break;
            }
        }
        xs.push(feats);
        ys.push(chosen as f32);
    }

    LearnedLinearEf::fit(&xs, &ys, *EF_LADDER.first().unwrap(), *EF_LADDER.last().unwrap())
        .expect("OLS fit failed — calibration set too small or degenerate")
}
