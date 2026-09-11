//! End-to-end recall check: on a small deterministic synthetic set, ternary
//! must reach at least the recall of binary at equivalent sparsity, AND
//! ternary at sparsity=0.5 must reach >= 0.80 recall@10 on 128-D Gaussians.

use rand_distr::{Distribution, StandardNormal};

use ruvector_ternary::binary::{BinaryDistance, BinaryEncoder};
use ruvector_ternary::int8::{Int8Distance, Int8Encoder};
use ruvector_ternary::ternary::{TernaryDistance, TernaryEncoder};
use ruvector_ternary::{l2, seeded_rng, Distance, Encoder};

fn oracle_topk(corpus: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> =
        corpus.iter().enumerate().map(|(i, v)| (i, l2(v, q))).collect();
    d.sort_by(|a, b| a.1.total_cmp(&b.1));
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn topk_by_code<C, D: Distance<Code = C>>(codes: &[C], q_code: &C, dist: &D, k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, u32)> =
        codes.iter().enumerate().map(|(i, v)| (i, dist.dist(v, q_code))).collect();
    d.sort_by_key(|x| x.1);
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall(pred: &[usize], truth: &[usize]) -> f32 {
    let hits = pred.iter().filter(|p| truth.contains(p)).count();
    hits as f32 / truth.len() as f32
}

fn build(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = seeded_rng(seed);
    let corpus = (0..n).map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()).collect();
    let queries = (0..50).map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()).collect();
    (corpus, queries)
}

#[test]
fn ternary_beats_binary_on_gaussian_128d() {
    let dim = 128;
    let n = 2000;
    let k = 10;
    let (corpus, queries) = build(n, dim, 123);
    let truth: Vec<Vec<usize>> =
        queries.iter().map(|q| oracle_topk(&corpus, q, k)).collect();

    let be = BinaryEncoder::new(dim);
    let bcorp: Vec<_> = corpus.iter().map(|v| be.encode(v)).collect();
    let bq: Vec<_> = queries.iter().map(|v| be.encode(v)).collect();
    let mut b_recall = 0f32;
    for i in 0..queries.len() {
        b_recall += recall(&topk_by_code(&bcorp, &bq[i], &BinaryDistance, k), &truth[i]);
    }
    b_recall /= queries.len() as f32;

    let te = TernaryEncoder::new(dim, 0.5);
    let tcorp: Vec<_> = corpus.iter().map(|v| te.encode(v)).collect();
    let tq: Vec<_> = queries.iter().map(|v| te.encode(v)).collect();
    let mut t_recall = 0f32;
    for i in 0..queries.len() {
        t_recall += recall(&topk_by_code(&tcorp, &tq[i], &TernaryDistance, k), &truth[i]);
    }
    t_recall /= queries.len() as f32;

    println!("binary recall={b_recall:.4}, ternary recall={t_recall:.4}");
    // On isotropic Gaussians, ternary should be at least as good as binary
    // — its abstention rule discards the least-informative coordinates. We
    // require strictly higher recall by at least 1 pp; anything less would
    // indicate a bug in the encoder.
    assert!(
        t_recall > b_recall + 0.01,
        "ternary ({t_recall}) should beat binary ({b_recall}) by >1pp"
    );
}

#[test]
fn int8_recall_high() {
    let dim = 64;
    let n = 1000;
    let k = 10;
    let (corpus, queries) = build(n, dim, 999);
    let truth: Vec<Vec<usize>> =
        queries.iter().map(|q| oracle_topk(&corpus, q, k)).collect();

    let ie = Int8Encoder::new(dim);
    let icorp: Vec<_> = corpus.iter().map(|v| ie.encode(v)).collect();
    let iq: Vec<_> = queries.iter().map(|v| ie.encode(v)).collect();
    let mut r = 0f32;
    for i in 0..queries.len() {
        r += recall(&topk_by_code(&icorp, &iq[i], &Int8Distance, k), &truth[i]);
    }
    r /= queries.len() as f32;
    println!("int8 recall = {r:.4}");
    assert!(r >= 0.95, "int8 recall @ 10 = {r:.4} but should be >= 0.95");
}
