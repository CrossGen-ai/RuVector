use super::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn gen_anisotropic(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Simulate data where subvector variance decays: dim j has stddev = 1/(1 + j/8)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0f32; n * d];
    let scales: Vec<f32> = (0..d).map(|j| 1.0 / (1.0 + j as f32 / 8.0)).collect();
    for i in 0..n {
        for j in 0..d {
            let normal = Normal::new(0.0, scales[j]).unwrap();
            out[i * d + j] = normal.sample(&mut rng) as f32;
        }
    }
    out
}

#[test]
fn pack_roundtrip() {
    let mut buf = vec![0u8; 16];
    let vals: [(u32, u8); 6] = [(5, 3), (10, 4), (200, 8), (1, 2), (63, 6), (7, 3)];
    let mut off = 0u32;
    for (v, b) in vals {
        pack::write_bits(&mut buf, off, b, v);
        off += b as u32;
    }
    let mut off = 0u32;
    for (v, b) in vals {
        let r = pack::read_bits(&buf, off, b);
        assert_eq!(r, v, "field bits={} off={}", b, off);
        off += b as u32;
    }
}

#[test]
fn water_fill_conservation() {
    let vars = vec![10.0, 5.0, 2.0, 1.0];
    let bits = water_fill::allocate_bits(&vars, 20, 2, 8).unwrap();
    assert_eq!(bits.iter().map(|&b| b as usize).sum::<usize>(), 20);
    // higher variance -> at least as many bits as lower
    for w in bits.windows(2) {
        // Not strictly monotone (because vars is monotone here and greedy is exact,
        // it will be non-increasing).
        assert!(w[0] >= w[1]);
    }
}

#[test]
fn water_fill_beats_uniform() {
    // Pick variances with a heavy tail so allocation matters.
    let vars = vec![50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25];
    let total = 32; // avg 4 bits per slot
    let bits = water_fill::allocate_bits(&vars, total, 2, 8).unwrap();
    let uniform = vec![4u8; 8];
    let d_ab = water_fill::predicted_distortion(&vars, &bits);
    let d_uni = water_fill::predicted_distortion(&vars, &uniform);
    assert!(d_ab < d_uni, "AB distortion {} should be < uniform {}", d_ab, d_uni);
    // meaningful improvement
    assert!(d_ab < 0.5 * d_uni, "expected >=2x improvement, got AB={} uni={}", d_ab, d_uni);
}

#[test]
fn end_to_end_encode_decode_shape() {
    let (n, d, m) = (400, 32, 8);
    let data = gen_anisotropic(n, d, 42);
    let cfg = AbPqConfig::new(d, m, 4);
    let pq = AbPq::train(cfg, &data).unwrap();
    assert_eq!(pq.bits.iter().map(|&b| b as usize).sum::<usize>(), 32);
    let codes = pq.encode_batch(&data);
    assert_eq!(codes.len(), n * pq.code_bytes());
    // ADC distance from a training vector should be small and non-negative
    let q = &data[0..d];
    let tbl = pq.build_adc_table(q);
    let d0 = pq.adc_distance(&codes[0..pq.code_bytes()], &tbl);
    assert!(d0 >= 0.0);
}

#[test]
fn abpq_lower_distortion_than_uniform_pq() {
    // Compare AB-PQ (variable bits, same total) vs uniform PQ trained identically.
    let (n, d, m) = (2000, 32, 8);
    let data = gen_anisotropic(n, d, 7);
    let mut cfg_ab = AbPqConfig::new(d, m, 4);
    cfg_ab.seed = 1;
    let ab = AbPq::train(cfg_ab, &data).unwrap();

    // Uniform baseline: force bit alloc = 4 everywhere by disabling budget play.
    let mut cfg_u = AbPqConfig::new(d, m, 4);
    cfg_u.min_bits = 4; cfg_u.max_bits = 4; cfg_u.seed = 1;
    let uni = AbPq::train(cfg_u, &data).unwrap();

    // Measure reconstruction MSE on data.
    let mse = |pq: &AbPq| -> f32 {
        let mut e = 0f32;
        let mut buf = vec![0u8; pq.code_bytes()];
        let dsub = d / m;
        for i in 0..n {
            let v = &data[i * d..(i + 1) * d];
            pq.encode_into(v, &mut buf);
            for s in 0..m {
                let code = pack::read_bits(&buf, pq.bit_offsets[s], pq.bits[s]) as u16;
                let cent = pq.subs[s].centroid(code);
                for j in 0..dsub {
                    let diff = v[s * dsub + j] - cent[j];
                    e += diff * diff;
                }
            }
        }
        e / n as f32
    };
    let e_ab = mse(&ab);
    let e_u = mse(&uni);
    println!("MSE ab={} uni={} ratio={:.3}", e_ab, e_u, e_ab / e_u);
    assert!(e_ab < e_u, "AB-PQ should beat uniform on anisotropic data");
}

#[test]
fn recall_at_10_at_least_matches_uniform() {
    let (n, d, m, nq) = (2000, 32, 8, 100);
    let data = gen_anisotropic(n, d, 3);
    let queries = gen_anisotropic(nq, d, 4);

    let mut cfg_ab = AbPqConfig::new(d, m, 4);
    cfg_ab.seed = 42;
    let ab = AbPq::train(cfg_ab, &data).unwrap();
    let mut cfg_u = AbPqConfig::new(d, m, 4);
    cfg_u.min_bits = 4; cfg_u.max_bits = 4; cfg_u.seed = 42;
    let uni = AbPq::train(cfg_u, &data).unwrap();

    let recall = |pq: &AbPq| -> f32 {
        let codes = pq.encode_batch(&data);
        let cb = pq.code_bytes();
        let mut hits = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            // ground truth: exact L2
            let mut gt: Vec<(f32, usize)> = (0..n).map(|i| {
                let v = &data[i * d..(i + 1) * d];
                let mut s = 0f32;
                for j in 0..d { let df = q[j] - v[j]; s += df * df; }
                (s, i)
            }).collect();
            gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt_set: std::collections::HashSet<usize> =
                gt.iter().take(10).map(|(_, i)| *i).collect();
            // ADC
            let tbl = pq.build_adc_table(q);
            let mut est: Vec<(f32, usize)> = (0..n).map(|i| {
                (pq.adc_distance(&codes[i * cb..(i + 1) * cb], &tbl), i)
            }).collect();
            est.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            hits += est.iter().take(10).filter(|(_, i)| gt_set.contains(i)).count();
        }
        hits as f32 / (nq * 10) as f32
    };
    let r_ab = recall(&ab);
    let r_u = recall(&uni);
    println!("recall@10 ab={} uni={}", r_ab, r_u);
    assert!(r_ab + 0.005 >= r_u, "AB-PQ recall {} regressed vs uniform {}", r_ab, r_u);
}
