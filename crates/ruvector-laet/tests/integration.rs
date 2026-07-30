use ruvector_laet::bench::{make_dataset, make_queries, run_strategy};
use ruvector_laet::features::Features;
use ruvector_laet::train::{fit_ridge, LinearModel};
use ruvector_laet::{build_training_set, ground_truth, FixedEf, Index, LaetStop, PatienceStop};

fn setup() -> (Index, Vec<Vec<f32>>, Vec<Vec<u32>>, Vec<u32>) {
    let data = make_dataset(5_000, 64, 42);
    let queries = make_queries(50, 64, 2002);
    let index = Index::new(data.clone(), 16);
    let stride = (index.graph.len() / 8).max(1) as u32;
    let entry_ids: Vec<u32> = (0..8).map(|i| i as u32 * stride).collect();
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| ground_truth(&data, q, 10)).collect();
    (index, queries, truths, entry_ids)
}

fn train_model(index: &Index, entry_ids: &[u32]) -> LinearModel {
    let train_q = make_queries(200, 64, 1001);
    let (xs, ys) = build_training_set(index, &train_q, entry_ids, 96, 10);
    let w = fit_ridge(&xs, &ys, 1e-2);
    let dummy = LinearModel { w: w.clone(), threshold: 0.0 };
    let mean: f32 = xs
        .iter()
        .map(|row| {
            let f = Features {
                iter: row[0],
                best_dist: row[1],
                delta_best_dist: row[2],
                iters_since_improve: row[3],
                ef_min_reached: row[4],
            };
            dummy.score(&f)
        })
        .sum::<f32>()
        / xs.len() as f32;
    LinearModel { w, threshold: mean * 0.15 }
}

#[test]
fn all_strategies_reach_recall_08() {
    let (index, queries, truths, entry_ids) = setup();
    let model = train_model(&index, &entry_ids);
    for row in [
        run_strategy(
            "fixed",
            &index,
            &queries,
            &truths,
            &entry_ids,
            10,
            FixedEf { ef: 64 },
        ),
        run_strategy(
            "patience",
            &index,
            &queries,
            &truths,
            &entry_ids,
            10,
            PatienceStop { patience: 10, max_iter: 64 },
        ),
        run_strategy(
            "laet",
            &index,
            &queries,
            &truths,
            &entry_ids,
            10,
            LaetStop { model: model.clone(), min_iter: 12, max_iter: 64 },
        ),
    ] {
        assert!(
            row.recall_at_10 >= 0.80,
            "strategy {} only reached recall {}",
            row.name,
            row.recall_at_10
        );
    }
}

#[test]
fn laet_uses_fewer_distance_calls_than_fixed_ef() {
    let (index, queries, truths, entry_ids) = setup();
    let model = train_model(&index, &entry_ids);
    let base = run_strategy(
        "fixed",
        &index,
        &queries,
        &truths,
        &entry_ids,
        10,
        FixedEf { ef: 64 },
    );
    let laet = run_strategy(
        "laet",
        &index,
        &queries,
        &truths,
        &entry_ids,
        10,
        LaetStop { model, min_iter: 12, max_iter: 64 },
    );
    assert!(
        laet.avg_dist_calls < base.avg_dist_calls,
        "expected LaetStop distance calls ({}) < FixedEf ({})",
        laet.avg_dist_calls,
        base.avg_dist_calls
    );
    assert!(
        (base.recall_at_10 - laet.recall_at_10) <= 0.05,
        "LaetStop recall regressed too far: base={} laet={}",
        base.recall_at_10,
        laet.recall_at_10
    );
}

#[test]
fn trainer_is_deterministic() {
    let (index, _, _, entry_ids) = setup();
    let a = train_model(&index, &entry_ids);
    let b = train_model(&index, &entry_ids);
    assert_eq!(a.w, b.w);
    assert!((a.threshold - b.threshold).abs() < 1e-6);
}
