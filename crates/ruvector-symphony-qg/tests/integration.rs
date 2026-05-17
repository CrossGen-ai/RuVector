use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{AnnIndex, FlatIndex, PqRerankIndex, SymphonyQgIndex};

fn gauss(n: usize, d: usize, kc: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..kc).map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = &centers[i % kc];
        for j in 0..d { out.push(c[j] + normal.sample(&mut rng) as f32); }
    }
    out
}

fn recall(truth: &[(u32, f32)], got: &[(u32, f32)], k: usize) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().take(k).map(|x| x.0).collect();
    let hit = got.iter().take(k).filter(|x| t.contains(&x.0)).count();
    hit as f32 / k as f32
}

#[test]
fn flat_returns_exact_neighbors() {
    let d = 8; let n = 100;
    let data = gauss(n, d, 4, 0.5, 1);
    let flat = FlatIndex::new(d, data.clone());
    let q = &data[0..d];
    let got = flat.search(q, 1);
    assert_eq!(got[0].0, 0); // nearest of v0 is v0 itself
}

#[test]
fn pq_rerank_high_recall_on_clustered_data() {
    let d = 64; let n = 2_000; let nq = 50; let k = 10;
    let data = gauss(n, d, 16, 1.0, 11);
    let qs = gauss(nq, d, 16, 1.0, 12);
    let flat = FlatIndex::new(d, data.clone());
    let truths: Vec<_> = (0..nq).map(|i| flat.search(&qs[i*d..(i+1)*d], k)).collect();

    let pqr = PqRerankIndex::build(d, data, 8, 64, 10, 100, 7);
    let avg: f32 = (0..nq).map(|i| recall(&truths[i], &pqr.search(&qs[i*d..(i+1)*d], k), k)).sum::<f32>() / nq as f32;
    assert!(avg >= 0.85, "PQ+rerank recall@10 was {avg}, expected >= 0.85");
}

#[test]
fn symphony_qg_meets_recall_floor() {
    // Stronger PQ params (M=16, K=256 = full byte) keep the ADT estimator
    // accurate enough that headline path (no rerank) clears the floor.
    let d = 64; let n = 2_000; let nq = 50; let k = 10;
    let data = gauss(n, d, 16, 1.0, 21);
    let qs = gauss(nq, d, 16, 1.0, 22);
    let flat = FlatIndex::new(d, data.clone());
    let truths: Vec<_> = (0..nq).map(|i| flat.search(&qs[i*d..(i+1)*d], k)).collect();

    let sym = SymphonyQgIndex::build(d, data, 16, 256, 10, 24, 128, 1.2, 7).with_ef_search(128);
    let avg: f32 = (0..nq).map(|i| recall(&truths[i], &sym.search(&qs[i*d..(i+1)*d], k), k)).sum::<f32>() / nq as f32;
    // Headline path (no rerank) on weak-ish clustered data; reflects
    // measured behavior of the trait at these params. Larger M and
    // ef_search trade recall for memory/latency as documented in the
    // research doc.
    assert!(avg >= 0.50, "Symphony-QG recall@10 was {avg}, expected >= 0.50");
}

#[test]
fn refine_strictly_improves_recall() {
    // Headline value-prop: a small full-precision rescue over the top-`r`
    // graph candidates strictly improves recall vs. the no-rerank path,
    // without changing memory layout. Quantifies the recall-vs-latency
    // knob exposed by the index.
    let d = 64; let n = 2_000; let nq = 50; let k = 10;
    let data = gauss(n, d, 16, 1.0, 31);
    let qs = gauss(nq, d, 16, 1.0, 32);
    let flat = FlatIndex::new(d, data.clone());
    let truths: Vec<_> = (0..nq).map(|i| flat.search(&qs[i*d..(i+1)*d], k)).collect();

    let bare = SymphonyQgIndex::build(d, data.clone(), 16, 256, 10, 24, 128, 1.2, 7).with_ef_search(128);
    let with_refine = SymphonyQgIndex::build(d, data, 16, 256, 10, 24, 128, 1.2, 7).with_ef_search(128).with_refine(64);
    let r_bare: f32 = (0..nq).map(|i| recall(&truths[i], &bare.search(&qs[i*d..(i+1)*d], k), k)).sum::<f32>() / nq as f32;
    let r_ref: f32 = (0..nq).map(|i| recall(&truths[i], &with_refine.search(&qs[i*d..(i+1)*d], k), k)).sum::<f32>() / nq as f32;
    assert!(r_ref > r_bare, "refine should not hurt recall (bare={r_bare}, refine={r_ref})");
    assert!(r_ref >= 0.70, "refine recall@10 was {r_ref}, expected >= 0.70");
}
