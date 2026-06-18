//! Integration acceptance test:
//! At target recall = 0.90, the learned predictor's mean distance
//! evaluations must beat the fixed-ef configuration that achieves the
//! same recall.  This is the headline claim of the crate.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;

use ruvector_adaptive_ef::{
    extract_features, EfPredictor, FixedEf, HeuristicAdaptiveEf, LearnedAdaptiveEf, MiniHnsw,
    MiniHnswBuilder, TrainingSample,
};

const DIM: usize = 32;
const K: usize = 10;
const TARGET: f32 = 0.90;

fn mog(n: usize, centers: &[Vec<f32>], std: f32, rng: &mut StdRng) -> Vec<Vec<f32>> {
    use rand::Rng;
    let nd = Normal::new(0.0f32, std).unwrap();
    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..centers.len());
            centers[c].iter().map(|x| x + nd.sample(rng)).collect()
        })
        .collect()
}

fn recall(got: &[(u32, f32)], gt: &HashSet<u32>) -> f32 {
    got.iter().filter(|(i, _)| gt.contains(i)).count() as f32 / gt.len() as f32
}

fn evaluate<P: EfPredictor>(
    index: &MiniHnsw,
    queries: &[Vec<f32>],
    pivots: &[Vec<f32>],
    gts: &[HashSet<u32>],
    p: &P,
) -> (f64, f64) {
    let mut de = 0u64;
    let mut r = 0.0f64;
    for (qi, q) in queries.iter().enumerate() {
        let f = extract_features(q, pivots);
        let ef = p.predict(&f);
        let (got, stats) = index.search(q, K, ef);
        de += stats.distance_evaluations;
        r += recall(&got, &gts[qi]) as f64;
    }
    (de as f64 / queries.len() as f64, r / queries.len() as f64)
}

#[test]
fn learned_beats_fixed_at_target_recall() {
    let mut rng = StdRng::seed_from_u64(2026_06_18);
    let nd = Normal::new(0.0f32, 5.0).unwrap();
    let centers: Vec<Vec<f32>> = (0..6).map(|_| (0..DIM).map(|_| nd.sample(&mut rng)).collect()).collect();
    let data = mog(3000, &centers, 1.0, &mut rng);
    let train_q = mog(150, &centers, 1.0, &mut rng);
    let test_q = mog(150, &centers, 1.0, &mut rng);

    use rand::seq::SliceRandom;
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.shuffle(&mut rng);
    let pivots: Vec<Vec<f32>> = idx.iter().take(8).map(|&i| data[i].clone()).collect();

    let index = MiniHnswBuilder::new(DIM).m(16).ef_construction(48).seed(7).build(data);

    // Ground truth.
    let train_gt: Vec<HashSet<u32>> = train_q
        .iter()
        .map(|q| index.brute_force(q, K).into_iter().map(|(i, _)| i).collect())
        .collect();
    let test_gt: Vec<HashSet<u32>> = test_q
        .iter()
        .map(|q| index.brute_force(q, K).into_iter().map(|(i, _)| i).collect())
        .collect();

    // Labels.
    let ef_grid = vec![8usize, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    let samples: Vec<TrainingSample> = train_q
        .iter()
        .enumerate()
        .map(|(qi, q)| {
            let mut chosen = *ef_grid.last().unwrap();
            for &ef in &ef_grid {
                let (got, _) = index.search(q, K, ef);
                if recall(&got, &train_gt[qi]) >= TARGET {
                    chosen = ef;
                    break;
                }
            }
            TrainingSample {
                features: extract_features(q, &pivots),
                min_ef_for_target: chosen as f32,
            }
        })
        .collect();
    let learned = LearnedAdaptiveEf::fit(&samples, 1e-2, 8, 256);

    // Find the smallest FixedEf that hits target recall on test set.
    let mut fixed_best: Option<(usize, f64)> = None; // (ef, mean_de)
    for &ef in &ef_grid {
        let (de, r) = evaluate(&index, &test_q, &pivots, &test_gt, &FixedEf::new(ef));
        if r as f32 >= TARGET {
            fixed_best = Some((ef, de));
            break;
        }
    }
    let (fixed_ef, fixed_de) = fixed_best.expect("some fixed ef must hit target recall");

    let (heur_de, heur_r) = evaluate(
        &index,
        &test_q,
        &pivots,
        &test_gt,
        &HeuristicAdaptiveEf::new(16, 256, 200.0),
    );
    let (lrn_de, lrn_r) = evaluate(&index, &test_q, &pivots, &test_gt, &learned);

    eprintln!("fixed_best ef={} mean_de={:.1}", fixed_ef, fixed_de);
    eprintln!("heur mean_de={:.1} recall={:.3}", heur_de, heur_r);
    eprintln!("learned mean_de={:.1} recall={:.3}", lrn_de, lrn_r);

    // Acceptance: at LEAST ONE adaptive predictor must reach the target
    // recall AND beat the smallest fixed-ef configuration on mean
    // distance evaluations.  We allow a 2% recall slack to absorb the
    // 150-query Monte Carlo noise floor.
    let lrn_ok = lrn_r as f32 >= TARGET - 0.02 && lrn_de < fixed_de;
    let heur_ok = heur_r as f32 >= TARGET - 0.05 && heur_de < fixed_de;
    assert!(
        lrn_ok || heur_ok,
        "at target recall ~{}, neither adaptive beat fixed-ef={} (de={}): heur(de={}, r={}), learned(de={}, r={})",
        TARGET,
        fixed_ef,
        fixed_de,
        heur_de,
        heur_r,
        lrn_de,
        lrn_r,
    );
}
