//! Integration tests for the three Quantizer variants.

use ruvector_anisotropic_pq::{
    dot, gen_gauss, rotation, sq_l2, AnisotropicPQ, AnisotropicPQR, Code, PlainPQ, PqParams,
    Quantizer,
};

fn params() -> PqParams {
    PqParams {
        dim: 32,
        m: 8,
        k: 16,
        iters: 12,
        seed: 42,
    }
}

#[test]
fn plain_pq_roundtrip_is_reasonable() {
    let data = gen_gauss(2000, 32, 8, 7);
    let q = PlainPQ::train(&data, params()).unwrap();
    let code = q.encode(&data[0]).unwrap();
    let rec = q.reconstruct(&code);
    let err = sq_l2(&data[0], &rec);
    assert!(err < 20.0, "plain pq recon err too high: {err}");
    assert_eq!(q.bytes_per_code(), 8);
}

#[test]
fn anisotropic_helps_mips_on_normalized_data() {
    let dim = 32;
    let n = 2000;
    let mut data = gen_gauss(n, dim, 8, 11);
    for v in &mut data {
        let n2: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        for a in v.iter_mut() {
            *a /= n2;
        }
    }
    let mut queries = gen_gauss(50, dim, 8, 12);
    for v in &mut queries {
        let n2: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        for a in v.iter_mut() {
            *a /= n2;
        }
    }
    let p = PqParams {
        dim,
        m: 8,
        k: 16,
        iters: 20,
        seed: 5,
    };
    let plain = PlainPQ::train(&data, p).unwrap();
    let aniso = AnisotropicPQ::train(&data, p, 4.0).unwrap();

    let plain_recall = mips_recall_reconstruct(&plain, &data, &queries, 10);
    let aniso_recall = mips_recall_reconstruct(&aniso, &data, &queries, 10);
    assert!(
        aniso_recall + 0.10 >= plain_recall,
        "aniso {aniso_recall} regressed vs plain {plain_recall}"
    );
}

#[test]
fn rotation_preserves_norm() {
    let data = gen_gauss(500, 32, 4, 3);
    let rot = rotation::Rotation::fit_variance_balancing(&data, 8, 1);
    for v in data.iter().take(20) {
        let r = rot.apply(v);
        let n0: f32 = v.iter().map(|a| a * a).sum();
        let n1: f32 = r.iter().map(|a| a * a).sum();
        assert!((n0 - n1).abs() < 1e-2, "rot changed norm: {n0} vs {n1}");
    }
}

#[test]
fn all_variants_encode_same_shape() {
    let data = gen_gauss(500, 32, 4, 9);
    let p = params();
    let a = PlainPQ::train(&data, p).unwrap();
    let b = AnisotropicPQ::train(&data, p, 3.0).unwrap();
    let c = AnisotropicPQR::train(&data, p, 3.0).unwrap();
    for v in data.iter().take(10) {
        assert_eq!(a.encode(v).unwrap().len(), 8);
        assert_eq!(b.encode(v).unwrap().len(), 8);
        assert_eq!(c.encode(v).unwrap().len(), 8);
    }
}

fn mips_recall_reconstruct<Q: Quantizer>(
    q: &Q,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let codes: Vec<Code> = data.iter().map(|v| q.encode(v).unwrap()).collect();
    let recon: Vec<Vec<f32>> = codes.iter().map(|c| q.reconstruct(c)).collect();
    let mut hit = 0usize;
    let mut total = 0usize;
    for qv in queries {
        let mut exact: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(qv, v)))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let gt: std::collections::HashSet<usize> =
            exact.into_iter().take(k).map(|(i, _)| i).collect();
        let mut approx: Vec<(usize, f32)> = recon
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(qv, v)))
            .collect();
        approx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, _) in approx.into_iter().take(k) {
            if gt.contains(&i) {
                hit += 1;
            }
        }
        total += k;
    }
    hit as f32 / total as f32
}
