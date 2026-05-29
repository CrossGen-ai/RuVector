use super::*;
use crate::search::{brute_force_topk, search, SearchStrategy};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn make_mixture(dim: usize, n: usize, n_blobs: usize, seed: u64) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(n_blobs);
    let center_noise = Normal::new(0.0, 4.0).unwrap();
    for _ in 0..n_blobs {
        let c: Vec<f32> = (0..dim).map(|_| center_noise.sample(&mut rng) as f32).collect();
        centers.push(c);
    }
    let blob_noise = Normal::new(0.0, 0.4).unwrap();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centers[i % n_blobs];
        for j in 0..dim {
            data.push(c[j] + blob_noise.sample(&mut rng) as f32);
        }
    }
    (data, centers)
}

#[test]
fn bet_matches_fullscan_recall() {
    let dim = 32;
    let n = 4_000;
    let (data, _) = make_mixture(dim, n, 32, 42);
    let mut rng = StdRng::seed_from_u64(7);
    let idx = IvfIndex::build(dim, data, 64, 8, &mut rng).unwrap();

    let (q, _) = make_mixture(dim, 50, 32, 99);
    let k = 10;
    let mut recall_sum = 0.0;
    let mut bet_scanned = 0usize;
    let mut full_scanned = 0usize;
    for qi in 0..50 {
        let query = &q[qi * dim..(qi + 1) * dim];
        let gt = brute_force_topk(&idx, query, k);
        let (bet, bs) = search(
            &idx,
            query,
            k,
            SearchStrategy::BoundedEarlyTerm {
                max_nprobe: 64,
                slack: 1.0,
            },
        );
        let (full, fs) = search(&idx, query, k, SearchStrategy::FixedNprobe(64));
        let gt_set: std::collections::HashSet<u32> = gt.iter().map(|(i, _)| *i).collect();
        let bet_set: std::collections::HashSet<u32> = bet.iter().map(|(i, _)| *i).collect();
        recall_sum += bet_set.intersection(&gt_set).count() as f32 / k as f32;
        bet_scanned += bs.vectors_scored;
        full_scanned += fs.vectors_scored;
        // soundness: BET must not lose vs full scan of all partitions
        for (i, (id, _)) in full.iter().enumerate() {
            assert_eq!(*id, bet[i].0, "BET soundness broke at qi={qi} rank={i}");
        }
    }
    let avg_recall = recall_sum / 50.0;
    assert!(avg_recall > 0.99, "recall {avg_recall} too low");
    assert!(bet_scanned < full_scanned, "BET must scan strictly less");
}

#[test]
fn fixed_budget_respects_budget() {
    let dim = 16;
    let n = 1_000;
    let (data, _) = make_mixture(dim, n, 16, 1);
    let mut rng = StdRng::seed_from_u64(2);
    let idx = IvfIndex::build(dim, data, 32, 5, &mut rng).unwrap();
    let q = vec![0.0f32; dim];
    let budget = 200;
    let (_, s) = search(&idx, &q, 10, SearchStrategy::FixedBudget(budget));
    // budget is enforced after each partition; allow overrun by at most one partition's last batch
    assert!(s.vectors_scored >= budget);
    let max_part = idx.partitions.iter().map(|p| p.members.len()).max().unwrap();
    assert!(s.vectors_scored <= budget + max_part);
}

#[test]
fn dim_mismatch_panics() {
    let dim = 8;
    let (data, _) = make_mixture(dim, 100, 4, 0);
    let mut rng = StdRng::seed_from_u64(0);
    let idx = IvfIndex::build(dim, data, 8, 3, &mut rng).unwrap();
    let q = vec![0.0f32; dim];
    let (_, _) = search(&idx, &q, 5, SearchStrategy::FixedNprobe(8));
}
