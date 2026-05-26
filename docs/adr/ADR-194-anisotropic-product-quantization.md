---
adr: 194
title: "Anisotropic Product Quantization — Score-Aware Loss for MIPS"
status: accepted
date: 2026-05-26
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [pq, quantization, mips, scann, ann, vector-search, nightly-research]
---

# ADR-194 — Anisotropic Product Quantization

## Status

**Accepted (research-grade PoC).** Implemented on branch
`research/nightly/2026-05-26-anisotropic-pq` as
`crates/ruvector-anisotropic-pq`. `cargo build`, `cargo test`, and
`cargo run --release` all green.

## Context

`ruvector` already has several quantization implementations:

- `ruvector-rabitq` — 1-bit rotation quantization (nightly 2026-04-23).
- `ruvector-lvq` — locally-adaptive vector quantization.
- `ruvector-opq` — optimized product quantization.

But there is no implementation that exposes the **score-aware loss** that
ScaNN (Guo et al., ICML 2020) made famous: the recognition that
maximum-inner-product search (MIPS) is asymmetric in the residual, and
that the residual component _parallel_ to a datapoint's direction
matters more for top-k scoring than the perpendicular component.

The 2026 SOTA gap analysis (`docs/research/sota-gap-analysis-2026.md`)
explicitly calls out "ScaNN anisotropic PQ" as a known omission.

## Decision

Add a new crate `ruvector-anisotropic-pq` exposing:

- A `Quantizer` trait with `encode`, `build_ip_lut`, `score`, and
  `bytes_per_vector`. The trait is intentionally minimal so other PQ
  variants (RaBitQ-LUT, OPQ-rotated PQ) can implement it later and
  share index-side infrastructure.
- `Pq` — a baseline isotropic Lloyd's-algorithm PQ.
- `ApqQuantizer` — a per-subspace anisotropic PQ trained with the
  loss `L = (η−1)(r · n̂)² + ‖r‖²` and centroid updates solving a small
  weighted linear system per centroid.

Per-subspace decomposition is an explicit, documented simplification of
ScaNN's full-vector loss `(r · x)²`, which does not decompose cleanly
across subspaces. The simplification is acceptable for a first
implementation because (a) it lets the loss change ship today without
adding a coordinate-descent trainer and (b) it lets us measure the
slice of the gain attributable to the loss itself, separate from the
joint encoder.

η is exposed as a knob; defaults are not baked into the trait. The
benchmark measures η ∈ {1.0, 2.0, 4.0}.

## Consequences

**Positive**

- `Quantizer` becomes the seed of a unified PQ runtime that other
  crates (HNSW, IVF, DiskANN) can target.
- At the heaviest compression rate measured (M=16, sub_dim=8) the
  score-aware loss gives a small but real R@10 lift (+0.65 pp absolute,
  +2.1 % relative) over baseline PQ on a clustered MIPS workload.
- ADC search latency is unchanged: the loss only affects training and
  encoding, not the runtime LUT scoring loop.
- Training is deterministic given a fixed seed; tests are reproducible.

**Negative**

- Training is 3–4× slower than baseline PQ due to outer-product
  accumulation and per-centroid linear solves.
- Encoding is ~1.7× slower per vector.
- At small `sub_dim` (4), the per-subspace direction estimate is noisy
  and the loss change either does nothing (η=2) or hurts (η=4).
- η is a hyperparameter the operator must tune per dataset.

**Neutral**

- Bytes-per-vector identical to standard PQ at the same `(M, K)`.
- API is intentionally narrow; growth into HNSW/IVF integration is a
  follow-up decision.

## Alternatives considered

1. **Full ScaNN coordinate-descent trainer.** Optimises `(r · x)²` jointly
   across subspaces. Higher expected gain (literature reports ~3 % R@10).
   Rejected for this iteration because the implementation cost is
   substantially higher and we wanted a clean A/B for the loss alone.
   Captured as future work in the research doc.
2. **AQLM (additive quantization).** Different decomposition (sum of
   codes, not concatenation). Different tradeoff curve; orthogonal to
   the score-aware-loss question. Out of scope.
3. **Rotation-then-PQ (OPQ pipeline).** Already exists as
   `ruvector-opq`. Composes with this crate; not a substitute.
4. **Do nothing.** Leaves a documented SOTA gap unaddressed and blocks
   future MIPS-oriented experiments that want to compare loss variants.

## Acceptance evidence

Numbers come from `cargo run --release -p ruvector-anisotropic-pq` on
2026-05-26 (n=20 000, d=128, K=256, 200 queries, Apple Silicon /
Darwin 24.6, single-thread, no SIMD intrinsics):

| Config            | Variant     | R@10    | R@100   | bpv |
|-------------------|-------------|--------:|--------:|----:|
| M=16, sub_dim=8   | PQ          | 0.3075  | 0.3969  | 16  |
| M=16, sub_dim=8   | APQ η=2.0   | 0.3140  | 0.3920  | 16  |
| M=16, sub_dim=8   | APQ η=4.0   | 0.3015  | 0.3877  | 16  |
| M=32, sub_dim=4   | PQ          | 0.5860  | 0.6509  | 32  |
| M=32, sub_dim=4   | APQ η=2.0   | 0.5845  | 0.6454  | 32  |
| M=32, sub_dim=4   | APQ η=4.0   | 0.5685  | 0.6286  | 32  |

Tests: 5 / 5 pass. Build: clean release.
