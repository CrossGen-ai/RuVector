//! ruvector-anisotropic-pq
//!
//! Anisotropic Product Quantization: ScaNN-style score-aware codebook
//! training that reduces error in the direction parallel to the datapoint,
//! which dominates inner-product / cosine score error. See
//! Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar, "Accelerating Large-Scale
//! Inference with Anisotropic Vector Quantization", ICML 2020.
//!
//! This crate provides three composable variants for direct A/B benchmark:
//!   - `LossKind::Reconstruction` — standard PQ (baseline).
//!   - `LossKind::Anisotropic { eta }` — score-aware loss with weight `eta`.
//!   - `LossKind::LearnedNorm { eta_min, eta_max }` — anisotropic with
//!     per-example weight scaled by ||x|| (empirically helps unnormalized data).
//!
//! No mocks, no stubs — full training and asymmetric distance (ADC) search.

pub mod data;
pub mod kmeans;
pub mod pq;

pub use pq::{AnisotropicPq, LossKind, PqConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_and_search_roundtrip() {
        let (data, _) = data::synthetic_gaussian(2000, 64, 8, 42);
        let queries = data::sample(&data, 64, 20, 1);
        let cfg = PqConfig {
            d: 64,
            m: 8,
            k: 16,
            iters: 8,
            loss: LossKind::Reconstruction,
            seed: 1,
        };
        let pq = AnisotropicPq::train(&cfg, &data);
        let codes = pq.encode_batch(&data);
        // Recall@10 must be > 20% even with tiny K=16 — sanity only.
        let recall = crate::pq::eval_recall(&pq, &data, &codes, &queries, 10);
        assert!(recall > 0.2, "recall={recall}");
    }

    #[test]
    fn anisotropic_and_baseline_both_produce_valid_mips_recall() {
        let (data, _) = data::synthetic_gaussian(4000, 64, 8, 7);
        // Normalize for MIPS = cosine
        let data = data::l2_normalize(&data, 64);
        let queries = data::sample(&data, 64, 40, 2);

        let base_cfg = PqConfig { d: 64, m: 8, k: 256, iters: 12, loss: LossKind::Reconstruction, seed: 11 };
        let ani_cfg  = PqConfig { d: 64, m: 8, k: 256, iters: 12, loss: LossKind::Anisotropic { eta: 4.0 }, seed: 11 };

        let base = AnisotropicPq::train(&base_cfg, &data);
        let ani  = AnisotropicPq::train(&ani_cfg,  &data);
        let cb = base.encode_batch(&data);
        let ca = ani.encode_batch(&data);
        // Anisotropic PQ optimizes inner-product / cosine (MIPS) score error,
        // not L2 reconstruction — evaluate on the metric it targets.
        use crate::pq::{eval_recall_metric, Metric};
        let r_base = eval_recall_metric(&base, &data, &cb, &queries, 10, Metric::Mips);
        let r_ani  = eval_recall_metric(&ani,  &data, &ca, &queries, 10, Metric::Mips);
        eprintln!("MIPS recall base={r_base:.3} aniso={r_ani:.3}");
        // Both variants should produce non-trivial recall. Note: on this
        // synthetic Gaussian corpus with L2-normalized vectors and K=256,
        // anisotropic training does NOT reliably beat the reconstruction
        // baseline in our reproducer — see docs/research/nightly/
        // 2026-07-21-anisotropic-pq/README.md for a detailed discussion of
        // when the ScaNN-style loss actually pays off and when it doesn't.
        assert!(r_base > 0.2, "baseline recall collapsed: {r_base}");
        assert!(r_ani  > 0.2, "anisotropic recall collapsed: {r_ani}");
    }
}
