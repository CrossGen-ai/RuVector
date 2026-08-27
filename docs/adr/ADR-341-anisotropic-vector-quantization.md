# ADR-341: Anisotropic Vector Quantization Codebooks for Inner-Product ANN

**Status:** Proposed

**Date:** 2026-08-27
**Owners:** ruvector maintainers
**Tracking:** [research/nightly/2026-08-27-anisotropic-vector-quantization](../research/nightly/2026-08-27-anisotropic-vector-quantization/README.md)

## Context

The repository already ships two families of compressed inner-product
scorers: `ruvector-pq-search` (classical MSE-trained product
quantization with IVF and residual variants) and `ruvector-rabitq`
(the VLDB'24 bit-quantizer with a proven recall bound). Both minimise a
symmetric MSE-style objective on the whole vector. Neither reshapes
the reconstruction error covariance to favour the parallel-to-query
component of the error, which is the term that actually shifts
inner-product scores.

Guo et al. (ScaNN, ICML 2020) showed a closed-form anisotropic loss
`h_∥ ‖e_∥‖² + h_⊥ ‖e_⊥‖²` with weight ratio derived from a target
inner-product threshold. The loss is dropped straight into Lloyd's
algorithm and produces PQ codebooks that dominate MSE-trained PQ on
the recall/QPS ladder for inner-product queries. The technique has
been the codebook default in Google's ScaNN since 2020 but is
absent from the ruvector line-up.

Nightly PoC evidence (see the research document) confirms two things
about porting the technique to the current codebase:

1. **On heteroskedastic-magnitude corpora, naïve AVQ regresses recall
   below the MSE baseline** (0.049 vs 0.090 at M=8, 0.086 vs 0.136 at
   M=16 on our synthetic corpus). This is not a bug — it is the
   directional loss being confounded by magnitude.
2. **Combining AVQ with unit-normalised training + a 4-byte per-vector
   norm side channel (`AvqNorm`) recovers and dramatically exceeds the
   MSE baseline** (0.468 at M=8, 0.446 at M=16), i.e. 3–5× recall
   improvement for +4 bytes per vector.

## Decision

Introduce `crates/ruvector-avq` as a standalone quantizer crate. Three
implementations of a single `Quantizer` trait: `PqMse` (baseline for
head-to-head evaluation), `AvqScoreAware` (paper as written), and
`AvqNorm` (the norm-decoupled variant that actually wins on our
benchmark). The crate is pure safe Rust, workspace-standard deps
(`rand`, `rand_distr`, `thiserror`, `serde`), and self-contained — no
dependency on `ruvector-core`, so it can be adopted by future crates
without dragging in the full ANN stack.

Ship it as *nightly research* rather than a production feature until
(a) it has been benchmarked on ANN-Benchmarks corpora (`glove-100-
angular`, `sift-1m`) and (b) it has a SIMD ADC kernel. The trait
boundary is designed so both extensions land without breaking
`AvqNorm`'s API.

Recommend `AvqNorm` — not plain `AvqScoreAware` — as the default when
inner-product recall matters and vectors are not already unit-
normalised. Recommend `PqMse` when magnitude is guaranteed uniform
(e.g. explicit L2-normalisation upstream), because in that regime the
score-aware bias adds no value.

## Consequences

* Adds one small crate (~1 000 LOC) to the workspace; compile time
  cost is negligible.
* Introduces the first codebook trainer in the repo whose loss is
  asymmetric between error components — sets a precedent other
  quantizers (RaBitQ, TurboVec, TurboQuant) can compose against.
* Establishes a reusable `Quantizer` trait boundary
  (`train` / `encode` / `adc` / `code_bytes` / `side_bytes`) that
  future variants (IVF+AVQ, additive quantization, SIMD ADC) can
  drop into without breaking downstream code.
* Publishes an explicit *practical failure mode*: naïve AVQ can
  regress recall on heteroskedastic-magnitude corpora. That
  documented finding is more valuable than a green benchmark table
  because it steers implementers away from a common trap.
* No production surface changes on `main` — nightly branch only,
  fork-only push. No PR to upstream.

## Alternatives Considered

* **Extend `ruvector-pq-search` in-place.** Rejected: it would mean
  either forking the existing trainer (churn) or making the loss
  pluggable through it (a bigger API commitment than a nightly PoC
  warrants). A separate crate is cheaper to iterate on and easier to
  delete if the recall win doesn't hold on real corpora.
* **Skip the AvqNorm variant and ship only the paper's algorithm.**
  Rejected on evidence: the numbers show plain AVQ regressing recall
  on this corpus. Shipping only the paper would ship a landmine.
* **Wait for a SIMD ADC before proposing anything.** Rejected: the
  interesting variable is the loss, not the kernel. A pure-scalar
  reference implementation is the right way to isolate the codebook
  effect from the kernel effect; SIMD can be a follow-up.
* **Adopt an external crate.** No existing Rust crate implements
  anisotropic-loss PQ that we could find on crates.io in 2026-08.
  Building it here also lets us keep the trait boundary aligned with
  the rest of `ruvector-*`.
