# ADR-361: Subspace-Conditional Product Quantization (SC-PQ)

- **Status**: Proposed (nightly research, 2026-11-17)
- **Deciders**: ruvector nightly research
- **Related**: ADR-* PQ variants (fastscan, anisotropic, dominant-code-skip), aq-beam-encode gist, rvq-cascade

## Context

Product Quantization (PQ) splits a D-dim vector into M subvectors and
quantizes each with an independent K-centroid codebook. Independence is
convenient — it gives the familiar LUT-based ADC scan — but it discards any
information carried by **cross-subspace correlation**. On embeddings whose
dimensions share latent factors (most learned embeddings do), two subspaces
are rarely jointly independent.

Prior remedies trade off scan simplicity: OPQ/anisotropic-PQ rotate the
space to approximate independence; RVQ cascades residual stages (more
memory, more scan compute); AQ / Beam-AQ learn fully-dependent additive
codebooks but incur a 2D LUT lookup that is exactly what SC-PQ makes cheap.

## Decision

Introduce **Subspace-Conditional PQ**: for each subspace m > 0 store K
sub-codebooks indexed by the previous subspace's chosen code. Training
buckets residuals per previous code; encoding is beam-search left-to-right
(beam=4 by default); the query-side LUT has shape `[M, K, K]`, and scan
cost is identical to classic PQ at M table lookups per vector.

Shipped as crate `ruvector-subspace-conditional-pq` with a trait
`Quantizer` so an index can A/B SC-PQ against the baseline PQ that ships
in the same crate.

## Consequences

Positive:

- Recall@10 improvement on correlated inputs in 3/4 tested regimes with
  **identical bit budget and scan throughput**. Measured on 2026-11-17:
  HIGH correlation 0.050 → 0.086 (+72%), M=4 0.020 → 0.070 (+250%),
  independent 0.040 → 0.086 (+115%).
- Scan throughput unchanged (~140 Mvec/s on this host, same as PQ).
- Backwards-compatible: the `Quantizer` trait keeps legacy PQ callers
  untouched.

Negative / open:

- LUT build is K× larger (K·K entries per subspace). At K=16 and M=8 this
  is 2 KiB per query; at K=32 it is 32 KiB and starts crowding L1.
- Codebook storage is K× the PQ codebook storage; still negligible
  (D·K·K floats).
- Encoding throughput drops ~4× at beam=4.
- One tested regime (MED correlation, M=8) **lost 1.4 recall points vs
  PQ**. SC-PQ should be benchmarked per dataset before adoption, and
  per-dataset selection (SC-PQ or PQ) can be made a build-time flag.
- Greedy encoding is unreliable; beam search is required (test
  `beam_search_improves_or_matches_greedy` enforces this invariant).

## Alternatives considered

- **OPQ before PQ** (ADR-* OPQ rotation gist): orthogonal rotation
  attacks the same correlation. SC-PQ is complementary — it handles
  correlation that remains after rotation.
- **AQ / Beam-AQ** (aq-beam-encode gist): fully additive codebooks give
  the strongest MSE but need an asymmetric 2D LUT per scan — strictly
  more work than SC-PQ's per-subspace 1D-indexed scan.
- **RVQ cascade** (rvq-cascade gist): residual stages reduce MSE but
  double memory per stage and add scan compute. SC-PQ keeps single-stage
  scan.

## Rollout

- Nightly branch only; no production call-sites yet.
- Suggested next step: wire SC-PQ into `ruvector-pq-search` behind a
  feature flag and A/B against PQ on sift1m-small and gist1m-small.
- Production crate layout: feature-gate `sc-pq` in
  `ruvector-pq-search`; expose both encoders via the existing
  `PqIndex` builder.
