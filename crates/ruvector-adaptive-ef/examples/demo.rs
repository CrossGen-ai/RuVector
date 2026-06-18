//! Tiny demo: builds a 2 000-point index, runs three predictors on a few
//! queries, prints the chosen `ef` and recall.
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;

use ruvector_adaptive_ef::{
    extract_features, EfPredictor, FixedEf, HeuristicAdaptiveEf, LearnedAdaptiveEf,
    MiniHnswBuilder, TrainingSample,
};

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let nd = Normal::new(0.0f32, 1.0).unwrap();
    let data: Vec<Vec<f32>> = (0..2000)
        .map(|_| (0..32).map(|_| nd.sample(&mut rng)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..5)
        .map(|_| (0..32).map(|_| nd.sample(&mut rng)).collect())
        .collect();
    let pivots: Vec<Vec<f32>> = data.iter().take(8).cloned().collect();

    let index = MiniHnswBuilder::new(32).m(12).ef_construction(48).build(data);

    // Fit a trivial learned model on the queries themselves (demo only).
    let labels: Vec<TrainingSample> = queries
        .iter()
        .map(|q| TrainingSample {
            features: extract_features(q, &pivots),
            min_ef_for_target: 32.0,
        })
        .collect();
    let learned = LearnedAdaptiveEf::fit(&labels, 1e-2, 8, 256);

    let preds: Vec<Box<dyn EfPredictor>> = vec![
        Box::new(FixedEf::new(64)),
        Box::new(HeuristicAdaptiveEf::new(16, 256, 50.0)),
        Box::new(learned),
    ];

    for (qi, q) in queries.iter().enumerate() {
        let gt: HashSet<u32> = index.brute_force(q, 10).into_iter().map(|(i, _)| i).collect();
        let f = extract_features(q, &pivots);
        println!("query {}: features={:?}", qi, f);
        for p in &preds {
            let ef = p.predict(&f);
            let (got, stats) = index.search(q, 10, ef);
            let hits = got.iter().filter(|(i, _)| gt.contains(i)).count();
            println!(
                "  {:<22} ef={:<4} de={:<5} recall@10={:.2}",
                p.name(),
                ef,
                stats.distance_evaluations,
                hits as f32 / 10.0
            );
        }
    }
}
