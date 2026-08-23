//! Integration test — SOAR must beat naive spilling (or tie) on recall@10 at
//! the same nprobe budget on a mixture dataset. If it doesn't, the assignment
//! rule is broken and CI fails.

use ruvector_soar::{
    brute_force, kmeans, recall_at_k,
    rng::Xor64, IvfNaiveSpill, IvfSoar, IvfTop1, PartitionIndex, Vector,
};

fn synth(n: usize, d: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xor64::new(seed);
    let cs: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..d).map(|_| rng.gauss() * 4.0).collect())
        .collect();
    (0..n).map(|i| {
        let c = &cs[i % clusters];
        (0..d).map(|k| c[k] + rng.gauss() * 0.6).collect()
    }).collect()
}

#[test]
fn soar_beats_or_matches_naive_spill_at_low_nprobe() {
    let n = 2000; let d = 32; let clusters = 16; let nlist = 64; let k = 10;
    let raw = synth(n, d, clusters, 111);
    let vecs: Vec<Vector> = raw.iter().enumerate()
        .map(|(i, v)| Vector { id: i as u32, data: v.clone() })
        .collect();
    let qs = synth(200, d, clusters, 222);
    let centroids = kmeans::train(&raw, nlist, 10, 333);

    let top1 = IvfTop1::build(vecs.clone(), centroids.clone());
    let spill = IvfNaiveSpill::build(vecs.clone(), centroids.clone(), 2);
    let soar = IvfSoar::build(vecs.clone(), centroids.clone(), 1.5);

    let gt: Vec<Vec<u32>> = qs.iter().map(|q| brute_force(&vecs, q, k)).collect();

    let mut r_top1 = 0f32; let mut r_spill = 0f32; let mut r_soar = 0f32;
    let np = 2usize;
    for (qi, q) in qs.iter().enumerate() {
        r_top1  += recall_at_k(&gt[qi], &top1.search(q, k, np));
        r_spill += recall_at_k(&gt[qi], &spill.search(q, k, np));
        r_soar  += recall_at_k(&gt[qi], &soar.search(q, k, np));
    }
    r_top1  /= qs.len() as f32;
    r_spill /= qs.len() as f32;
    r_soar  /= qs.len() as f32;
    eprintln!("recall @nprobe=2: top1={r_top1:.4} spill={r_spill:.4} soar={r_soar:.4}");

    // 1. Spilling backends must exceed baseline.
    assert!(r_spill > r_top1, "spill {r_spill} vs top1 {r_top1}");
    assert!(r_soar  > r_top1, "soar {r_soar} vs top1 {r_top1}");

    // 2. SOAR must be at least as good as naive spill (typical: strictly better).
    assert!(r_soar >= r_spill - 0.005,
        "soar {r_soar} should meet or beat naive spill {r_spill}");
}
