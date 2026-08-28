# ADR-340: FINGER-style low-rank distance approximation for graph ANN

- **Status**: Proposed (nightly research, 2026-08-28)
- **Deciders**: ruvector nightly research pipeline
- **Related**: nightly research 2026-04-23 (rabitq), 2026-06-16 (coherence-hnsw), 2026-06-25 (capability-gated ANN), 2026-06-21 (matryoshka-coarse-fine)
- **Crate**: `crates/ruvector-finger`
- **Branch**: `research/nightly/2026-08-28-finger-hnsw-distance`

## Context

`ruvector-coherence-hnsw`, `ruvector-acorn`, `ruvector-diskann` and the
speculative/diverse-beam variants all inherit HNSW's core hot loop: **scoring
`M` neighbours of the current best node against the query**. For d = 128 to 1024,
that is `M * d` FMA operations per hop. Existing knobs (RaBitQ, PQ-ADC,
Matryoshka) attack the *neighbour vector* representation but still pay a
per-neighbour projection cost.

FINGER (Yin et al., WWW 2023, *"Fast Inference for Graph-based Approximate
Nearest Neighbour Search"*) makes a different observation: **most neighbours
you score during graph traversal share a parent pivot**, so you can precompute
a low-rank residual basis *per pivot* and reduce each neighbour-score to a
short dot product plus one shared anchor term. It is orthogonal to all our
existing quantisers and can be layered on top of them.

We already ship half a dozen graph-ANN crates; none of them cache per-pivot
approximation state today. We need to decide whether FINGER is worth the
memory footprint (rank*4 bytes per vector) and the per-pivot PCA cost at
build/update time.

## Decision

Land `ruvector-finger` as a stand-alone crate that exposes a
`DistanceEstimator` trait with three interchangeable implementations:

1. **`ExactEstimator`** -- baseline `q.v` (single-precision).
2. **`JlEstimator<r>`** -- global Gaussian JL projection of residuals (dumb
   low-rank baseline).
3. **`FingerEstimator<r>`** -- per-pivot PCA basis built via deflated power
   iteration.

The trait is deliberately HNSW-agnostic so we can wire it into
`ruvector-coherence-hnsw`, `ruvector-acorn`, `ruvector-diskann`, and the
diverse-beam/speculative variants without touching their graph code.

`ruvector-finger` is not automatically enabled anywhere; each consumer opts
in via feature flag once the recall/latency budget has been validated on
that graph.

## Consequences

**Positive**
- 1.4x to 2.2x speedup on the inner scoring loop with zero recall loss
  (r=16) or up to 3% recall loss (r=8). See Benchmarks section for real numbers.
- 8x to 16x reduction in per-vector state when FINGER is used as the primary
  scorer (still fall back to exact for the final rerank).
- Orthogonal to existing quantisation crates -- can stack on top of RaBitQ,
  Matryoshka or PQ.

**Negative**
- Per-pivot PCA has O(P * M * d * iters) build cost. Measured 887 ms for
  n=20 000, d=128, P=141, r=16 on M4 Max, single-thread. Amortised.
- Per-pivot basis storage: `P * r * d * 4` bytes. For 100k pivots at
  d=1024, r=16 that is 6.4 GB -- mitigated by capping pivot count in the
  consumers (SPANN/IVF style).
- Updates require re-running PCA for the affected pivot; suits the
  `lsm-ann` / `spann-partition-spill` batching model, not raw upserts.

**Neutral**
- Approximation quality depends on residual intrinsic dimension. Our
  coherence embeddings and delta-index summaries have effective rank
  under 32, well within the r=16 regime we validated.

## Alternatives considered

- **Stay with RaBitQ / PQ-ADC only.** Both quantise the *vector*; they do
  not amortise a *shared* per-pivot projection across `M` neighbours. FINGER
  composes with them rather than replacing them.
- **DeepSeed / learned entry points.** Reduces the number of hops but not
  the cost per hop. Complementary; not exclusive.
- **CAGRA-style GPU graph search.** Off-limits for this branch (Rust CPU
  target), but the FINGER trick applies there too.
- **Full-dimensional residual storage.** Removes the low-rank
  approximation error but costs the same as keeping the vector.

## Benchmarks

See `docs/research/nightly/2026-08-28-finger-hnsw-distance/README.md`.

Hardware: Apple M4 Max, arm64, single-thread, release mode, `criterion`
default sampler (20 samples, 2 s measurement, 1 s warm-up).

Per-pivot batch scoring (average 71 neighbours, d=128):

| estimator   | time     | speedup vs exact | bytes/vec |
|-------------|----------|------------------|-----------|
| exact       | 1.96 us  | 1.00x            | 512       |
| jl-r16      | 1.61 us  | 1.22x            |  64       |
| finger-r16  | 1.38 us  | 1.42x            |  64       |
| finger-r8   | 0.87 us  | 2.24x            |  32       |

End-to-end query loop (n=20 000, d=128, 200 queries, beam=4, rerank=200,
top-k=10, exact rerank on shortlist):

| estimator   | per-query | recall@10 |
|-------------|-----------|-----------|
| exact       | 54.9 us   | 0.364     |
| jl-r16      | 44.7 us   | 0.316     |
| finger-r8   | 31.9 us   | 0.353     |
| finger-r16  | 35.0 us   | 0.364     |

FINGER-r16 achieves recall parity with exact at 64% of the wall-clock cost
and 12.5% of the memory. FINGER-r8 loses 3% recall for 2.24x scoring
throughput.
