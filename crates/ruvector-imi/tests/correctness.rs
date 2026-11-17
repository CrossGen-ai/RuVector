//! Correctness: with a sufficient shortlist IMI must converge toward
//! exact-NN on clustered data; k-means must produce non-empty cells on
//! average; multi-sequence enumeration must not double-visit a cell.

use ruvector_imi::data::{make_clustered, make_queries};
use ruvector_imi::kmeans::l2_sq;
use ruvector_imi::{AnnIndex, FlatIvf, ImiIndex};

fn exact_top1(data: &[f32], d: usize, q: &[f32]) -> u32 {
    let n = data.len() / d;
    let mut best = 0u32;
    let mut best_sq = f32::INFINITY;
    for i in 0..n {
        let sq = l2_sq(q, &data[i * d..(i + 1) * d]);
        if sq < best_sq {
            best_sq = sq;
            best = i as u32;
        }
    }
    best
}

#[test]
fn imi_converges_to_exact_with_large_shortlist() {
    let n = 5_000;
    let d = 32;
    let data = make_clustered(n, d, 128, 0.05, 1);
    let queries = make_queries(50, d, 128, 0.05, 2);
    let mut imi = ImiIndex::build(data.clone(), d, 16, 15, 7);
    imi.shortlist_cap = n; // scan everything via multi-seq

    let mut hit = 0;
    for q in 0..50 {
        let query = &queries[q * d..(q + 1) * d];
        let truth = exact_top1(&data, d, query);
        let got = imi.search(query, 1 << 20, 1);
        if got[0].0 == truth {
            hit += 1;
        }
    }
    // With shortlist == n every point is evaluated, recall must be 100%.
    assert_eq!(hit, 50, "full-shortlist IMI must equal exact NN");
}

#[test]
fn flat_ivf_full_probe_equals_exact() {
    let n = 2_000;
    let d = 16;
    let data = make_clustered(n, d, 64, 0.1, 3);
    let queries = make_queries(30, d, 64, 0.1, 4);
    let k = 32;
    let ivf = FlatIvf::build(data.clone(), d, k, 10, 11);
    for q in 0..30 {
        let query = &queries[q * d..(q + 1) * d];
        let truth = exact_top1(&data, d, query);
        let got = ivf.search(query, k, 1);
        assert_eq!(got[0].0, truth, "probe=K must equal exact");
    }
}

#[test]
fn imi_cells_sum_to_n() {
    let n = 4_000;
    let d = 16;
    let data = make_clustered(n, d, 64, 0.1, 5);
    let imi = ImiIndex::build(data, d, 16, 10, 9);
    let total: usize = imi.cells.iter().map(|c| c.len()).sum();
    assert_eq!(total, n, "every vector must land in exactly one cell");
}
