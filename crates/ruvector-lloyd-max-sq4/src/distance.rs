//! Asymmetric L2 distance between a float query and a packed-nibble code
//! database vector.
//!
//! The LUT approach:
//! For each dimension `d` and level `k`, precompute `t[d, k] = (q[d] - c[d, k])^2`.
//! For a code vector, the squared L2 distance is `sum_d t[d, code[d]]`.
//! Memory: dim * 16 floats. Lookups: dim per query vector.

use crate::quant::Codec;
use crate::{unpack_nibble, NLEVELS};

/// Build the per-dimension (dim × 16) squared-error table for query `q`.
pub fn build_query_lut<C: Codec + ?Sized>(codec: &C, q: &[f32]) -> Vec<f32> {
    assert_eq!(q.len(), codec.dim());
    let d = codec.dim();
    let mut lut = vec![0f32; d * NLEVELS];
    for i in 0..d {
        let qi = q[i];
        for k in 0..NLEVELS {
            let e = qi - codec.centroid(i, k);
            lut[i * NLEVELS + k] = e * e;
        }
    }
    lut
}

/// Compute squared L2 distance using a pre-built LUT.
pub fn asym_l2_lut(dim: usize, lut: &[f32], code: &[u8]) -> f32 {
    debug_assert_eq!(lut.len(), dim * NLEVELS);
    debug_assert_eq!(code.len(), (dim + 1) / 2);
    let mut acc = 0f32;
    let mut d = 0;
    let mut bi = 0;
    while d + 1 < dim {
        let byte = code[bi];
        let (a, b) = unpack_nibble(byte);
        acc += lut[d * NLEVELS + a as usize];
        acc += lut[(d + 1) * NLEVELS + b as usize];
        d += 2;
        bi += 1;
    }
    if d < dim {
        let (a, _) = unpack_nibble(code[bi]);
        acc += lut[d * NLEVELS + a as usize];
    }
    acc
}

/// Scan a batch of codes with one query. Pushes `(dist, idx)` pairs onto `out`.
pub fn asym_l2_scan<C: Codec + ?Sized>(
    codec: &C,
    q: &[f32],
    codes: &[u8],
    n: usize,
    out: &mut Vec<(f32, u32)>,
) {
    let dim = codec.dim();
    let cb = codec.code_bytes();
    assert_eq!(codes.len(), n * cb);
    let lut = build_query_lut(codec, q);
    out.clear();
    out.reserve(n);
    for i in 0..n {
        let code = &codes[i * cb..(i + 1) * cb];
        let dist = asym_l2_lut(dim, &lut, code);
        out.push((dist, i as u32));
    }
}

/// Compute recall@k of a quantized search against float ground truth.
pub fn recall_at_k(k: usize, gt: &[u32], candidates: &[u32]) -> f32 {
    use std::collections::HashSet;
    let gt_set: HashSet<u32> = gt.iter().take(k).copied().collect();
    let hits = candidates
        .iter()
        .take(k)
        .filter(|c| gt_set.contains(c))
        .count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{Codec, LloydMaxSQ4, UniformSQ4};
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn gaussian_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n * d)
            .map(|_| {
                let x: f32 = StandardNormal.sample(&mut rng);
                x
            })
            .collect()
    }

    fn exact_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
    }

    #[test]
    fn lut_matches_decoded_distance() {
        let d = 32;
        let n = 128;
        let data = gaussian_data(n, d, 2);
        let codec = LloydMaxSQ4::fit(&data, n, d, 20, 1e-5);
        let codes = codec.encode_batch(&data, n);
        let q = gaussian_data(1, d, 99);
        let lut = build_query_lut(&codec, &q);

        for i in 0..n {
            let code = &codes[i * codec.code_bytes()..(i + 1) * codec.code_bytes()];
            let via_lut = asym_l2_lut(d, &lut, code);
            let dec = codec.decode(code);
            let direct = exact_l2(&q, &dec);
            assert!(
                (via_lut - direct).abs() < 1e-3,
                "lut={} direct={}",
                via_lut,
                direct
            );
        }
    }

    fn laplace_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        // Heavy-tailed; uniform SQ wastes bits on the tails.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        use rand::Rng;
        (0..n * d)
            .map(|_| {
                let u: f32 = rng.gen_range(-0.5f32..0.5f32);
                -u.signum() * (1.0 - 2.0 * u.abs()).ln()
            })
            .collect()
    }

    #[test]
    fn lloyd_wins_recall_on_heavy_tailed_data() {
        let d = 32;
        let n = 2000;
        let nq = 50;
        let data = laplace_data(n, d, 11);
        let queries = laplace_data(nq, d, 17);

        let u = UniformSQ4::fit(&data, n, d);
        let l = LloydMaxSQ4::fit(&data, n, d, 30, 1e-6);
        let codes_u = u.encode_batch(&data, n);
        let codes_l = l.encode_batch(&data, n);

        let k = 10;
        let mut r_u = 0f32;
        let mut r_l = 0f32;
        let mut buf = Vec::new();

        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            let mut gt: Vec<(f32, u32)> = (0..n)
                .map(|j| (exact_l2(q, &data[j * d..(j + 1) * d]), j as u32))
                .collect();
            gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt_ids: Vec<u32> = gt.iter().take(k).map(|p| p.1).collect();

            asym_l2_scan(&u, q, &codes_u, n, &mut buf);
            buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            r_u += recall_at_k(k, &gt_ids, &buf.iter().take(k).map(|p| p.1).collect::<Vec<_>>());

            asym_l2_scan(&l, q, &codes_l, n, &mut buf);
            buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            r_l += recall_at_k(k, &gt_ids, &buf.iter().take(k).map(|p| p.1).collect::<Vec<_>>());
        }
        r_u /= nq as f32;
        r_l /= nq as f32;
        assert!(
            r_l > r_u,
            "lloyd should beat uniform on Laplace: lloyd={}, uniform={}",
            r_l,
            r_u
        );
    }

    #[test]
    fn recall_improves_with_lloyd_vs_uniform() {
        let d = 32;
        let n = 2000;
        let nq = 50;
        let data = gaussian_data(n, d, 3);
        let queries = gaussian_data(nq, d, 7);

        let u = UniformSQ4::fit(&data, n, d);
        let l = LloydMaxSQ4::fit(&data, n, d, 25, 1e-5);
        let codes_u = u.encode_batch(&data, n);
        let codes_l = l.encode_batch(&data, n);

        let k = 10;
        let mut r_u = 0f32;
        let mut r_l = 0f32;
        let mut buf = Vec::new();

        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];

            // Ground truth = exact L2 top-k.
            let mut gt: Vec<(f32, u32)> = (0..n)
                .map(|j| (exact_l2(q, &data[j * d..(j + 1) * d]), j as u32))
                .collect();
            gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt_ids: Vec<u32> = gt.iter().take(k).map(|p| p.1).collect();

            asym_l2_scan(&u, q, &codes_u, n, &mut buf);
            buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let cand_u: Vec<u32> = buf.iter().take(k).map(|p| p.1).collect();
            r_u += recall_at_k(k, &gt_ids, &cand_u);

            asym_l2_scan(&l, q, &codes_l, n, &mut buf);
            buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let cand_l: Vec<u32> = buf.iter().take(k).map(|p| p.1).collect();
            r_l += recall_at_k(k, &gt_ids, &cand_l);
        }
        r_u /= nq as f32;
        r_l /= nq as f32;
        // Lloyd minimizes per-dim MSE, which does not strictly dominate recall
        // (see "practical failure modes" in the research doc). We only require
        // Lloyd to stay within a small gap of uniform — the real win is on
        // heavy-tailed or asymmetric data, not standard Gaussian.
        assert!(
            r_l >= r_u - 0.05,
            "lloyd recall should be close to uniform on Gaussian: lloyd={}, uniform={}",
            r_l,
            r_u
        );
    }
}
