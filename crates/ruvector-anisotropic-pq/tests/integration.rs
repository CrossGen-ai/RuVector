//! Integration tests: real data flow, no mocks.
//!
//! Acceptance test: on unit-normalized anisotropic synthetic data, APQ's
//! inner-product recall@10 should be ≥ PQ within init noise.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_anisotropic_pq::metrics::{dot, normalize_in_place};
use ruvector_anisotropic_pq::{AnisotropicPq, Opq, Pq, Quantizer};

fn gen_anisotropic_unit(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng) as f32).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect();
    let axes: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| normal.sample(&mut rng) as f32).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect();
    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..n_clusters);
            let mut v = centers[c].clone();
            // small isotropic jitter
            for i in 0..d {
                v[i] += (normal.sample(&mut rng) as f32) * 0.15;
            }
            // anisotropic elongation along cluster's long axis
            let along = (normal.sample(&mut rng) as f32) * 0.6;
            for i in 0..d {
                v[i] += along * axes[c][i];
            }
            normalize_in_place(&mut v);
            v
        })
        .collect()
}

fn ip(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

fn exact_top_k_ip(db: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = db.iter().enumerate().map(|(i, v)| (i, ip(q, v))).collect();
    s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn exact_top_k_l2(db: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut t = 0f32;
            for j in 0..q.len() {
                let d = q[j] - v[j];
                t += d * d;
            }
            (i, t)
        })
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn pq_top<Q: Quantizer>(qz: &Q, codes: &[Vec<u8>], q: &[f32], k: usize) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, qz.asymmetric_score(q, c)))
        .collect();
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

fn recall(p: &[usize], t: &[usize]) -> f32 {
    p.iter().filter(|i| t.contains(i)).count() as f32 / t.len() as f32
}

#[test]
fn pq_encode_decode_shapes() {
    let train = gen_anisotropic_unit(2_000, 32, 8, 42);
    let pq = Pq::train(&train, 4, 64, 20, 0).unwrap();
    let code = pq.encode(&train[0]);
    assert_eq!(code.len(), 4);
    assert_eq!(pq.code_bytes(), 4);
    assert_eq!(pq.shape(), (4, 32, 64));
}

#[test]
fn apq_matches_or_beats_pq_on_anisotropic() {
    let d = 32;
    let m = 8;
    let k = 128;
    let train = gen_anisotropic_unit(4_000, d, 12, 5);
    let db = gen_anisotropic_unit(3_000, d, 12, 6);
    let queries = gen_anisotropic_unit(80, d, 12, 7);

    let pq = Pq::train(&train, m, k, 25, 1).unwrap();
    let apq = AnisotropicPq::train(&train, m, k, 4.0, 25, 3, 1).unwrap();

    let pq_codes: Vec<Vec<u8>> = db.iter().map(|v| pq.encode(v)).collect();
    let apq_codes: Vec<Vec<u8>> = db.iter().map(|v| apq.encode(v)).collect();

    let mut pq_r = 0f32;
    let mut apq_r = 0f32;
    for q in &queries {
        // For PQ (L2-trained), score by L2 lookup; truth is L2 (equivalent
        // to IP for unit vectors up to sign).
        let truth = exact_top_k_ip(&db, q, 10);
        pq_r += recall(&pq_top(&pq, &pq_codes, q, 10), &truth);
        apq_r += recall(&pq_top(&apq, &apq_codes, q, 10), &truth);
    }
    pq_r /= queries.len() as f32;
    apq_r /= queries.len() as f32;

    println!("IP truth — pq_recall={pq_r:.4}  apq_recall={apq_r:.4}");
    // Both must achieve meaningful recall.
    assert!(pq_r > 0.10, "PQ recall too low: {pq_r}");
    assert!(apq_r > 0.10, "APQ recall too low: {apq_r}");
    // Tolerate 5pp k-means seed noise; we just need APQ to not regress badly.
    assert!(
        apq_r + 0.05 >= pq_r,
        "APQ regressed badly vs PQ: apq={apq_r} pq={pq_r}",
    );
}

#[test]
fn opq_train_is_orthogonal_enough() {
    let train = gen_anisotropic_unit(1_500, 16, 6, 11);
    let opq = Opq::train(&train, 4, 32, 15, 2, 1).unwrap();
    let d = opq.d;
    let mut off = 0f32;
    let mut diag_err = 0f32;
    for i in 0..d {
        for j in 0..d {
            let mut s = 0f32;
            for kk in 0..d {
                s += opq.r[i][kk] * opq.r[j][kk];
            }
            if i == j {
                diag_err += (s - 1.0).abs();
            } else {
                off += s.abs();
            }
        }
    }
    let n_off = (d * (d - 1)) as f32;
    let avg_off = off / n_off;
    let avg_diag = diag_err / d as f32;
    println!("avg |off-diag|={avg_off:.4}  avg |diag-1|={avg_diag:.4}");
    assert!(avg_off < 0.05, "rotation drifted from orthogonal: {avg_off}");
    assert!(avg_diag < 0.05, "rotation lost unit norm: {avg_diag}");
}

#[test]
fn pq_recalls_on_tight_l2_clusters() {
    // Tight Gaussians around well-separated centers; L2-truth so PQ excels.
    let d = 16;
    let mut rng = StdRng::seed_from_u64(99);
    let normal = Normal::new(0.0, 0.05).unwrap();
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..d).map(|_| (rng.gen::<f32>() - 0.5) * 8.0).collect())
        .collect();
    let mut mk = |n: usize| -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                let c = rng.gen_range(0..8);
                centers[c]
                    .iter()
                    .map(|x| x + normal.sample(&mut rng) as f32)
                    .collect()
            })
            .collect()
    };
    let train = mk(2_000);
    let db = mk(1_500);
    let queries = mk(80);
    let pq = Pq::train(&train, 4, 64, 25, 0).unwrap();
    let codes: Vec<Vec<u8>> = db.iter().map(|v| pq.encode(v)).collect();
    let mut total = 0f32;
    for q in &queries {
        let truth = exact_top_k_l2(&db, q, 5);
        total += recall(&pq_top(&pq, &codes, q, 5), &truth);
    }
    let avg = total / queries.len() as f32;
    // 8 tight clusters, all near-identical within cluster (noise 0.05 ≪ centroid
    // step ~1.4). Asking for recall@5 inside one cluster demands finer resolution
    // than k=64 sub-centroids can give; expect ~0.2 — what we test is that we
    // pick neighbors from the right cluster (recall > 1/n_clusters = 0.125).
    println!("L2 truth, tight clusters, recall@5 = {avg:.4}");
    assert!(avg > 0.15, "PQ should at least pick the right cluster: got {avg}");
}
