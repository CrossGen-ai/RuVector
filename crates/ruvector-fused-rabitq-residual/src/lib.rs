//! Fused RaBitQ + 4-bit Scalar Residual Quantization.
//!
//! This crate implements a two-stage quantizer that combines the geometric
//! guarantees of RaBitQ (1-bit sign codes over a rotated basis, ADR-260)
//! with a 4-bit scalar-quantized residual that captures per-dimension
//! deviation from the pure-sign reconstruction.
//!
//! See `docs/research/nightly/2026-09-04-fused-rabitq-residual/README.md`
//! for the SOTA survey, design rationale, and benchmark results.

pub mod fused;
pub mod quantizer;
pub mod rabitq;
pub mod rotation;
pub mod scan;
pub mod sq4;

pub use fused::FusedRQR;
pub use quantizer::{QueryCtx, Quantizer};
pub use rabitq::RabitQuant;
pub use rotation::SignedHadamard;
pub use scan::QuantizedIndex;
pub use sq4::Sq4Quant;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};

    fn gaussian_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn rotation_preserves_norm() {
        let d = 128;
        let rot = SignedHadamard::new_seeded(d, 7);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..50 {
            let v: Vec<f32> = (0..d)
                .map(|_| rng.gen_range(-2.0..2.0))
                .collect();
            let n0: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut vr = v.clone();
            rot.apply(&mut vr);
            let n1: f32 = vr.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (n0 - n1).abs() < 1e-4 * n0.max(1.0),
                "norm not preserved: {n0} vs {n1}"
            );
        }
    }

    #[test]
    fn code_sizes_match_analytic_formula() {
        let d = 128;
        let rabit = RabitQuant::new(d, 0);
        let sq4 = Sq4Quant::new(d, 0);
        let fused = FusedRQR::new(d, 0);
        assert_eq!(rabit.code_bytes(), 4 + d / 8); // 20
        assert_eq!(sq4.code_bytes(), 8 + d / 2); // 72
        assert_eq!(fused.code_bytes(), 12 + d / 8 + d / 2); // 92
    }

    fn recall_and_mse<Q: Quantizer>(
        q: Q,
        vecs: &[Vec<f32>],
        queries: &[Vec<f32>],
        ground: &[Vec<usize>],
        k: usize,
    ) -> (f32, f32) {
        let idx = QuantizedIndex::build(q, vecs);
        // reconstruction MSE
        let mut mse = 0.0f64;
        let mut count = 0usize;
        for (i, v) in vecs.iter().enumerate().take(64) {
            let code = &idx.codes[i * idx.code_len..(i + 1) * idx.code_len];
            let ctx = idx.q.prepare_query(v); // rotate v as if it were a query
            let d = idx.q.distance(code, &ctx);
            mse += d as f64;
            count += 1;
            let _ = i;
        }
        let mse = (mse / count as f64) as f32;

        let mut recall = 0.0f32;
        for (qi, query) in queries.iter().enumerate() {
            let topk = idx.topk(query, k);
            let got: std::collections::HashSet<usize> =
                topk.iter().map(|(i, _)| *i).collect();
            let want: std::collections::HashSet<usize> =
                ground[qi].iter().take(k).cloned().collect();
            recall += got.intersection(&want).count() as f32 / k as f32;
        }
        (recall / queries.len() as f32, mse)
    }

    fn exact_topk(vecs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s = v
                    .iter()
                    .zip(q.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                (i, s)
            })
            .collect();
        d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        d.iter().take(k).map(|(i, _)| *i).collect()
    }

    #[test]
    fn fused_beats_rabitq_alone_on_reconstruction() {
        let d = 128;
        let n = 2000;
        let m = 20;
        let vecs = gaussian_vectors(n, d, 42);
        let queries = gaussian_vectors(m, d, 43);
        let ground: Vec<Vec<usize>> =
            queries.iter().map(|q| exact_topk(&vecs, q, 10)).collect();

        let (r_rabit, mse_rabit) =
            recall_and_mse(RabitQuant::new(d, 7), &vecs, &queries, &ground, 10);
        let (r_fused, mse_fused) =
            recall_and_mse(FusedRQR::new(d, 7), &vecs, &queries, &ground, 10);
        let (r_sq4, mse_sq4) =
            recall_and_mse(Sq4Quant::new(d, 7), &vecs, &queries, &ground, 10);

        eprintln!(
            "recall  rabit={r_rabit:.3}  sq4={r_sq4:.3}  fused={r_fused:.3}"
        );
        eprintln!(
            "self-mse rabit={mse_rabit:.4}  sq4={mse_sq4:.4}  fused={mse_fused:.4}"
        );

        // Fused must strictly reduce reconstruction MSE vs pure RaBitQ.
        assert!(
            mse_fused < mse_rabit * 0.5,
            "fused MSE not << RaBitQ MSE: {mse_fused} vs {mse_rabit}"
        );
        // Fused recall must beat pure RaBitQ recall by a clear margin.
        assert!(
            r_fused >= r_rabit + 0.05,
            "fused recall not > RaBitQ recall by margin: {r_fused} vs {r_rabit}"
        );
        // Fused recall must be within noise of SQ4 (typically higher; we
        // assert not-worse-by-much so the test is stable across seeds).
        assert!(
            r_fused >= r_sq4 - 0.05,
            "fused recall regressed vs SQ4: {r_fused} vs {r_sq4}"
        );
    }
}
