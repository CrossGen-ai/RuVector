# ADR-357: μ-law Companded Scalar Quantization for Embeddings

Status: Proposed (nightly research PoC, 2026-11-03)

## Context

RuVector supports several scalar-quantization (SQ) codecs — linear int8, LVQ,
Lloyd–Max SQ4, Hadamard SQ, variance-adaptive SQ, per-cluster SQ. All treat
the per-dimension value distribution as approximately uniform within a
clipped range. For dense embeddings coming out of transformer encoders,
that is a decent fit on average but a bad fit per-dimension: most components
cluster tightly around zero with a long thin tail that drives the clipping
scale.

Classical signal-processing solved this problem with **companding** —
compress on encode, expand on decode — long before anyone thought to
quantize word embeddings. ITU-T G.711 μ-law and A-law are the standard
examples: both telephony and PCM audio spent a decade coding 13-bit linear
speech into 8-bit non-uniform codes with no audible loss, because they
devote more bins where the signal spends most of its time.

μ-law companding has not, as of this writing, been applied to vector-search
quantization in any shipped engine we are aware of. The research doc
surveys the few mentions in the literature.

## Decision

Add a `mulaw_sq8` scalar-quantization codec as an experimental alternative
to `linear_sq8` in RuVector, with a `fit_learned_mu(train, clip_q)` helper
that picks μ by matching the median absolute value to half the code range.
Expose it behind the same `ScalarCodec` trait so IVF / HNSW re-rank paths
can swap codecs with no index-structure changes.

Keep `linear_sq8` as the default: μ-law only wins when the empirical
value distribution has true heavy tails (e.g. unnormalized BERT CLS, some
multimodal encoders). On rotation-flattened or hadamard-pre-whitened data
the two codecs tie, and linear is faster.

## Consequences

**Positive**

- 4× memory reduction identical to linear SQ8 — no index-side changes.
- When μ is learned from the data, recall@100 matches float32 within
  0.5 pp on 128-D Gaussian embeddings (0.9883 vs 0.9881 fp32-reproduced
  via linear) and reconstruction MSE matches linear SQ8 to within noise
  (1.81e-4 vs 1.83e-4).
- μ-law is a pointwise, dimension-independent transform, so it composes
  trivially with Hadamard/OPQ rotations and with existing IVF/HNSW code.

**Negative / open**

- Naïve encode/decode uses `ln`/`exp`, slowing throughput ~15× vs linear
  SQ8 in the current PoC (2.7 Mvec/s encode vs 31 Mvec/s). The fix is a
  256-entry LUT — the full inverse table fits in L1 and recovers linear
  speed. Deferred to the production crate.
- Fixed μ=255 (the G.711 default) is wrong for Gaussian-tailed embeddings
  and visibly hurts recall (0.9704 vs 0.9786 R@10). μ must be fit from
  data; shipping without a fit routine would be a trap.

## Alternatives Considered

- **A-law** (G.711 piecewise-linear companding). Faster to implement without
  LUT, but less expressive and the piecewise discontinuities hurt gradient-
  based codebook refinement if we ever add one.
- **Lloyd–Max SQ** (already in `gist-lloyd-max-sq4-2026`). Optimal in MSE but
  per-codebook storage, slower to fit, and does not inherit the smooth,
  parametric shape that lets μ-law be re-fit online with a one-pass stat.
- **LVQ / variance-adaptive SQ.** Also already shipped. These adapt scale
  per-dimension; μ-law adapts the *shape* within a single scale. The two
  are orthogonal — the natural next step is μ-law + LVQ.
- **Hadamard pre-rotation + linear SQ.** Already shipped. Flattens the
  per-dimension distribution so linear is near-optimal. On already-rotated
  data μ-law has nothing to add, which is why we are not making it default.

## Links

- Research doc: [docs/research/nightly/2026-11-03-mulaw-sq-companding/README.md](../research/nightly/2026-11-03-mulaw-sq-companding/README.md)
- PoC crate: [crates/mulaw-sq-companding/](../../crates/mulaw-sq-companding/)
