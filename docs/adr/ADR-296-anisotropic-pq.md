# ADR-296: Anisotropic Product Quantization for MIPS

- **Status**: Proposed (nightly research prototype)
- **Date**: 2026-08-05
- **Deciders**: RuVector Research (nightly automation)
- **Related**: ADR-265 (PQ-ADC), ADR-264 (Matryoshka), the `ruvector-pq-search` and `ruvector-rabitq` crates
- **Tags**: pq, mips, quantization, scann, recall

## Context

RuVector's isotropic PQ-ADC pipeline (`ruvector-pq-search`) minimises
sub-space L2 reconstruction error. For **inner-product** queries — the
dominant workload for LLM retrieval, ANCE / E5 / BGE embeddings, and
RAG systems — reconstruction error is not the same as ranking error:
residuals aligned with the datum's own direction directly perturb the
inner-product score, while orthogonal residuals average out.

ScaNN (Guo et al., ICML 2020) showed that up-weighting the parallel
residual in the k-means loss ("anisotropic vector quantization") lifts
top-k recall at fixed code size by several percentage points on GloVe
and BERT-family corpora. RuVector has PQ, RaBitQ, and Matryoshka but no
score-aware PQ variant — this ADR closes that gap.

## Decision

Land a new workspace crate `ruvector-anisotropic-pq` that provides:

1. A `PqCodebookTrainer` trait with two implementations,
   `IsotropicTrainer` (baseline, equivalent to standard Lloyd's k-means)
   and `AnisotropicTrainer { eta }` (weighted Lloyd's with per-centroid
   `d × d` linear solve).
2. An `AnisoPqIndex` that consumes any trainer and performs an ADC top-k
   search whose query-time code path is byte-identical to `FlatPqIndex`
   in `ruvector-pq-search`.
3. A `aniso-pq-bench` binary that reports build time, per-query latency,
   recall@10, and memory bytes across an η-sweep for reproducibility.

The trainer choice is a build-time selection; downstream callers using
PQ indexes are not required to change their query code.

## Consequences

**Positive**

- **Higher recall at zero query-cost delta.** Measured +4.2 pp
  recall@10 at η=2 on a low-rank + noise 8 192 × 64 corpus, code size
  and per-query µs unchanged.
- **Trainer swap is one line.** `PqCodebookTrainer` trait keeps the
  index generic; existing `FlatPqIndex` / `IvfPqIndex` consumers can
  adopt anisotropic training when they upgrade.
- **Pure safe Rust, no BLAS.** Sub-vector `d ≤ 32` keeps the per-centroid
  Gauss-Jordan solve cheap; the crate has zero non-workspace
  dependencies beyond `rand` and `thiserror`.

**Negative / risk**

- **~2.5× slower training.** Weighted Lloyd's iteration solves a `d×d`
  linear system per centroid update; training is offline and small in
  absolute terms (~1.5 s for 8 k vectors) but should be measured on
  million-scale corpora before default-on rollout.
- **Corpus-dependent benefit.** Anisotropic PQ under-performs on
  purely isotropic-Gaussian data with no covariance structure — the
  nightly bench documents this failure mode explicitly.
- **η is a hyper-parameter.** The recall curve is unimodal; recommend
  η ∈ [2, 8] with cross-validation. A follow-up ADR can address
  per-sub-space adaptive η.

## Alternatives considered

- **Extend `ruvector-pq-search` in place.** Rejected: keeps the isotropic
  baseline and the anisotropic variant coupled, complicating the crate's
  narrow "reference PQ" charter. A new crate documents the new loss
  cleanly and can graduate into `ruvector-pq-search` once field-tested.
- **RaBitQ only.** 1-bit quantization is a different Pareto trade-off
  (much smaller codes, hardware-friendly Hamming distance) and is
  already shipped. Anisotropic PQ targets a different regime
  (medium-compression, ranking-critical MIPS).
- **OPQ (learned rotation) instead of anisotropic loss.** Complementary,
  not competing; the follow-up work item explicitly stacks the two.

## Follow-ups

- Wire into `ruvector-sota-bench` for SIFT-1M / GloVe-1M / MS-MARCO
  numbers.
- Full-vector direction with per-sub-space projection (closer to the
  ScaNN paper's derivation).
- Adaptive per-sub-space η selection.
- SIMD LUT + `f16` centroids for query-throughput lift.
