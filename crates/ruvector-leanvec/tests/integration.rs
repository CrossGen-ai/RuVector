//! Integration test: end-to-end recall sanity on a small anisotropic corpus.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_leanvec::{FlatIndex, LeanVecIndex, LvqIndex, Projection, VectorIndex};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = (d / 8).max(2);
    let mut basis = vec![0.0_f32; latent * d];
    for k in 0..latent {
        for j in 0..d {
            basis[k * d + j] = rng.gen::<f32>() - 0.5;
        }
        let mut s = 0.0;
        for j in 0..d {
            s += basis[k * d + j] * basis[k * d + j];
        }
        let inv = 1.0 / s.sqrt();
        for j in 0..d {
            basis[k * d + j] *= inv;
        }
    }
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        let row = &mut out[i * d..(i + 1) * d];
        for k in 0..latent {
            let coef = (rng.gen::<f32>() - 0.5) * 5.0;
            for j in 0..d {
                row[j] += coef * basis[k * d + j];
            }
        }
        for j in 0..d {
            row[j] += (rng.gen::<f32>() - 0.5) * 0.05;
        }
    }
    out
}

fn recall_at_k(
    flat: &FlatIndex,
    other: &dyn VectorIndex,
    queries: &[f32],
    d: usize,
    k: usize,
) -> f64 {
    let nq = queries.len() / d;
    let mut hits = 0;
    let mut total = 0;
    for i in 0..nq {
        let q = &queries[i * d..(i + 1) * d];
        let f = flat.search(q, k);
        let g = other.search(q, k);
        let truth: std::collections::HashSet<u32> = f.iter().map(|n| n.id).collect();
        for n in g {
            if truth.contains(&n.id) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f64 / total as f64
}

#[test]
fn lvq_and_leanvec_keep_high_recall_on_anisotropic_data() {
    let n = 1_500;
    let d = 64;
    let r = 32;
    let k = 10;
    let db = synth(n, d, 11);
    let queries = synth(80, d, 12);

    let mut flat = FlatIndex::new(d);
    let mut lvq = LvqIndex::new(d);
    let proj = Projection::train_pca(&db, n, d, r, 13);
    let mut lv = LeanVecIndex::new(proj, 4);

    for i in 0..n {
        let v = &db[i * d..(i + 1) * d];
        flat.add(v);
        lvq.add(v);
        lv.add(v);
    }

    let r_lvq = recall_at_k(&flat, &lvq, &queries, d, k);
    let r_lv = recall_at_k(&flat, &lv, &queries, d, k);

    // LVQ-8 is essentially loss-free at this scale.
    assert!(r_lvq >= 0.95, "LVQ recall {r_lvq} below 0.95");
    // LeanVec with rerank should be ≥ 0.90 even with r = d/2; the rerank
    // stage on retained f32 originals is the safety net.
    assert!(r_lv >= 0.90, "LeanVec recall {r_lv} below 0.90");
}

#[test]
fn memory_savings_in_expected_band() {
    // For d=128, LVQ-8 alone gives ~3.8× shrink (128 bytes + 8 overhead vs
    // 512). LeanVec(r=d/2) + retained f32 originals is roughly equal in
    // memory to flat — it trades memory back for projection-domain speed.
    let n = 500;
    let d = 128;
    let r = 64;
    let db = synth(n, d, 21);

    let mut flat = FlatIndex::new(d);
    let mut lvq = LvqIndex::new(d);
    let proj = Projection::train_pca(&db, n, d, r, 22);
    let mut lv = LeanVecIndex::new(proj, 4);
    for i in 0..n {
        let v = &db[i * d..(i + 1) * d];
        flat.add(v);
        lvq.add(v);
        lv.add(v);
    }

    let flat_bv = flat.bytes() as f64 / n as f64;
    let lvq_bv = lvq.bytes() as f64 / n as f64;
    let lv_bv = lv.bytes() as f64 / n as f64;
    assert!(
        lvq_bv > 0.20 * flat_bv && lvq_bv < 0.30 * flat_bv,
        "LVQ bytes/vec {lvq_bv} not in 20-30% of flat {flat_bv}"
    );
    // LeanVec retains f32 originals plus codes: ~1.13× flat.
    assert!(
        lv_bv > flat_bv && lv_bv < 1.25 * flat_bv,
        "LeanVec bytes/vec {lv_bv} unexpected vs flat {flat_bv}"
    );
}
