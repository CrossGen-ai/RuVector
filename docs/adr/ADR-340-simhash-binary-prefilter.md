# ADR-340: SimHash Binary Prefilter for ANN Candidate Reduction

**Status:** Accepted (research spike)

**Date:** 2026-08-25
**Owners:** ruvector maintainers
**Tracking:** nightly branch `research/nightly/2026-08-25-simhash-prefilter-ann`

## Context

`ruvector` already carries several quantized-index crates —
`ruvector-rabitq` (rotated 1-bit quantization with a proven error bound),
`ruvector-pq-search` (product quantization), `ruvector-turboquant` (LUT
quant). Each is a **full index**: it owns storage and search.

A different, and orthogonal, primitive is missing: a **pre-index binary
prefilter** that any existing index can bolt on the front of its exact
distance step. Concretely, an HNSW or brute-force search whose distance
kernel is float32 L2 wants to cheaply drop candidates before it pays the
`dim × 4` byte cache-line load per comparison.

The question this ADR settles: **does a classical Charikar-style SimHash
prefilter, kept intentionally minimal (no rotation, no learning), give
enough recall-preserving pruning to be worth its ~8-32 bytes/vector, and
should it live as a trait-based crate distinct from the existing
quantized indices?**

## Decision

Yes — introduce **`crates/ruvector-simhash-prefilter/`**, a
trait-based, index-agnostic SimHash prefilter with `#![forbid(unsafe_code)]`.

Shape:

- `SketchFamily<const W: usize>` trait so future backends (OPQ-rotated,
  learned, super-bit) plug in without churning downstream indices.
- Const-generic `Sketch<W>` for compile-time monomorphised popcount at
  64, 128 and 256 bits.
- `SrpFamily<W>` — the concrete Rademacher signed-random-projection
  implementation, deterministic given a seed.
- `FlatPrefilterIndex<F, W>` — a reference brute-force index that
  composes the prefilter with an exact L2 rerank. Not the intended
  production consumer, but the reference implementation and the
  benchmark harness.

The crate lives as its own workspace member (not a feature flag on an
existing quantized crate) because:

1. It is **not a full index** — it produces candidate ids and defers
   distance to a caller-owned reranker. Merging it into
   `ruvector-rabitq` would blur that contract.
2. Its dependency surface is minimal (`rand`, `rand_distr`,
   `thiserror`, workspace `rayon` on non-wasm) and matches the pattern
   already used by other single-purpose crates like
   `ruvector-visited-filter` and `ruvector-hnsw-repair`.
3. Downstream crates (a future `ruvector-simhash-hnsw`) can consume
   the trait and add HNSW-specific glue without pulling any index
   internals.

## Consequences

**Positive.**

- On a 20 000×128 clustered corpus, the 64-bit prefilter at candidate
  multiplier 40 achieves **100% recall@10 in 60 µs**, an **11.5×
  speedup** over the 693 µs exact scan, with only **8 bytes per
  vector** of prefilter storage (a 64× shrink vs. the raw
  512-byte vector). Numbers from `cargo run --release -p
  ruvector-simhash-prefilter` — see the research doc for the full
  table.
- The `SketchFamily` trait defers the "which projection is best"
  question. RaBitQ-style rotation, learned sketches, super-bit LSH all
  fit behind the same interface.
- Zero unsafe. Portable popcount via `u64::count_ones` compiles to
  `popcntq` on x86_64 and `cnt.8b` on aarch64.
- Adds one workspace member, one binary, one benchmark. Compiles
  cleanly with `cargo build --release -p ruvector-simhash-prefilter`;
  `cargo test -p ruvector-simhash-prefilter` runs 5 unit tests including
  a recall-monotonicity property.

**Negative / risks.**

- **The wins are corpus-dependent.** Isotropic Gaussian queries on
  isotropic Gaussian corpora give ~12% recall@10 (the anti-pattern is
  documented in the research doc). Real learned embeddings sit
  closer to the clustered regime we measured, but heterogeneous
  workloads can regress.
- **Projection matrix persistence is a new failure mode.** Losing the
  seed poisons every stored sketch. The API forces the caller to hold
  the family; a future serialisation wrapper should record the seed
  alongside the sketch blob.
- **Candidate multiplier is a live tuning knob.** Callers who pick
  mult=10 for latency will see recall dips at some corpora; the
  benchmark output makes this visible but does not eliminate the need
  for per-corpus tuning.
- **Sketch encode cost is O(bits × dim).** At `dim ≥ 1536`, the encoder
  becomes non-trivial; the prefilter only wins above ~2 000 vectors at
  those dimensions.

## Alternatives considered

1. **Extend `ruvector-rabitq` with a "sketch-only" mode.** Rejected —
   RaBitQ is a full quantized index with an error bound; a raw
   Charikar sketch is a different contract (no bound, no distance
   estimate, just a Hamming rank).
2. **Extend `ruvector-visited-filter`.** Rejected — that crate is a
   per-query bitset gating HNSW visits, not a persistent per-vector
   sketch.
3. **Ship only as an HNSW feature flag.** Rejected — the primitive is
   independently useful (brute-force pre-scan, IVF cell reduction) and
   coupling it to HNSW would foreclose those consumers.
4. **Skip the SketchFamily trait; hardcode SRP.** Rejected — the trait
   is 8 lines and the swap-in cost is real: RaBitQ, super-bit LSH,
   learned sketches all belong behind it.
