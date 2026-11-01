# ADR-355: Asymmetric RaBitQ (2-bit query / 1-bit database)

- **Status**: Proposed
- **Date**: 2026-11-01
- **Deciders**: RuVector Nightly Research
- **Related**: ADR-357 (spherical-pvq), earlier RaBitQ work
- **Tags**: quantization, rabitq, binary-ann, recall, no-storage-cost

## Context

RaBitQ-style 1-bit quantization keeps a vector as the sign pattern of its
Hadamard-rotated coordinates. Database storage is one bit per dimension
(16 bytes for `dim=128`), scoring is a POPCNT, and recall at a fixed
rerank depth is capped by the Hamming estimator's variance.

Two orthogonal ways to push recall further are:

1. **Store more bits per database vector** — multi-bit RaBitQ, Extended
   RaBitQ, LVQ, SQ4. All of these inflate storage.
2. **Store more bits per *query* only** — the database stays 1 bit, the
   query spends more bits because a query is encoded once per lookup, not
   once per record. In an ANN scan the asymmetric cost sits on the query
   side where it is amortized across `N` dot products.

Approach (2) is the "asymmetric distance" trick that FAISS uses for PQ
(ADC vs SDC). RaBitQ's symmetric variant discards it. Measurable on this
machine, a ternary (sign + mask) query over a 1-bit database improves
recall@10 by ~51% at identical database storage and essentially identical
scan time.

## Decision

Add a feature-gated backend `AsymQ2D1` to the quantization family:

- Database codes stay unchanged: 1 bit/dim, Hadamard-rotated sign.
- Queries are encoded as a ternary vector: `sign` and `mask` bits per dim
  (~1.58 bits/dim). The mask is driven by a per-query threshold `τ` set
  at a configurable fraction of the median absolute component.
- The estimator is `||q||·||d||·(1 − 2·H/active)` where `H` is the number
  of mismatches among `active` masked dimensions. For `τ=0`, this
  recovers the symmetric RaBitQ estimator exactly.
- The rotation and codebook are shared between all backends so a single
  index can serve F32Flat / SymRabitQ / AsymQ2D1 side by side.

The quality bar for landing in a production crate is:
- Recall@10 strictly higher than symmetric at the same scan time, on
  Gaussian and real embeddings.
- No change to the on-disk database layout.
- Pure CPU, no NEON/AVX intrinsics in the first cut.

## Consequences

**Pros.** No storage increase. No index rebuild — the same bit codes
serve both estimators and the choice is per-query. Query encoding adds
one O(dim) sort for the median; this is dominated by the index scan for
realistic `N`. Trivially combines with existing re-ranking stages (ADC,
exact float refinement, SOAR spill).

**Cons.** Query-side CPU grows slightly (one extra POPCNT lane and an
AND). The mask threshold `τ` is a new tunable; recall is sensitive to it,
so a validation pass must choose it per dataset.

## Alternatives considered

- **Multi-bit RaBitQ (2 bits per DB dim).** Doubles database storage —
  rejected when the win is a cheap recall lift.
- **Extended RaBitQ (B-bit).** Same objection.
- **Full ADC rerank over flat floats.** Already available as a
  downstream stage; this ADR is about the fast front-end, not the
  rerank.
- **PQ fastscan.** Different code family; complementary, not a
  replacement.

## Acceptance

- `cargo build --release -p ruvector-asymmetric-rabitq-q2d1` green.
- `cargo test --release -p ruvector-asymmetric-rabitq-q2d1` passes the
  recall non-regression assertion.
- Demo binary prints measured recall@10 and per-query latency for all
  three backends on the same corpus.
