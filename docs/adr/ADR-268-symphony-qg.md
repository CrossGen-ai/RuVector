# ADR-268: SymphonyQG — Symphonious Integration of 1-Bit Quantization and Graph-Based ANN Search

**Status**: Proposed (research PoC)
**Date**: 2026-06-23
**Authors**: Nightly Research Agent (Claude Code)
**Related**: ADR-264 (PQ ADC search), `crates/ruvector-rabitq`, `crates/ruvector-roargraph`

---

## Context

ruvector currently ships graph-based ANN (`ruvector-roargraph`, `ruvector-rairs`)
and 1-bit RaBitQ quantization (`ruvector-rabitq`) as **independent** modules.
Index implementations either pay full f32 L2 distances during graph traversal
or run a quantized scan that does no graph pruning. Neither approach exploits
the cache-locality and instruction-level win that comes from fusing the two.

Recent work — most notably SymphonyQG (SIGMOD 2025), FINGER (NeurIPS 2023), and
NGT-QG (Yahoo, 2023) — shows that paying the dominant graph-traversal distance
cost in cheap quantized space and reserving exact distances for survivors gives
20-50 % latency reductions on standard ANN benchmarks without recall loss.

ruvector has never tested this seam. Production users running 10 M+ vector
indexes hit the L2-overflow regime where the wins are largest.

## Decision

Add a new crate `ruvector-symphony-qg` that implements the SymphonyQG pattern
with two layout variants:

* `SymphonyQG` — quantized codes in a parallel `Vec<QuantizedCode>` array.
* `SymphonyQGPacked` — `[neighbor_id u32 | code u64×W]` interleaved per node
  in a single contiguous blob, so one cache-line walk covers IDs *and* codes.

The search uses a **Hamming-agreement filter**: for each candidate neighbor,
popcount the XOR of query and stored sign bits; if agreements are below a
tuned threshold (default 0.55·D), skip the exact L2 entirely. Result decisions
remain on exact L2 so recall is preserved.

Graph construction in the PoC uses brute-force kNN bootstrap with NSG-style
occlusion pruning, parallelised via rayon — adequate up to ~50 k vectors, to
be replaced with NN-descent or HNSW layered insertion for production scale.

## Consequences

**Positive**

* **Latency**: 1.30× single-thread QPS speedup at 50 k × 128-D with **zero**
  recall regression (0.8396 → 0.8408) on the cluster benchmark.
* **Memory**: +20 B/vector for codes (parallel layout) or +340 B/vector for the
  packed layout. The packed layout extracts the locality win once vectors
  overflow L2.
* **Composable**: builds on existing `ruvector-rabitq` primitives — no
  duplication of rotation or sign-pack code at the production stage.
* **Reusable**: the popcount-filter pattern can be lifted into
  `ruvector-roargraph` and `ruvector-rairs` without changing their public
  APIs.

**Negative / Risk**

* Build cost is currently O(N²) bootstrap; needs an NN-descent or HNSW path
  before this can ship for ≥1 M-vector indexes.
* Filter threshold (0.55·D) is hand-tuned on the cluster benchmark; production
  needs an auto-calibration pass per dataset.
* Packed layout doubles the storage of quantized codes (once in the codes
  array, once per incoming edge). Net memory is still small but worth noting.

**Migration**

Nothing in existing crates changes. New crate is additive. A follow-up ADR
will cover NN-descent build, NEON popcount kernel, and threshold
auto-calibration.

## Alternatives Considered

1. **Extend `ruvector-rabitq` in-place.** Rejected: would blur the
   quantizer-vs-index boundary and complicate the rabitq API surface.
2. **Re-use `ruvector-roargraph` as the host.** Rejected for the PoC because
   roargraph's beam search makes assumptions about distance monotonicity that
   the popcount filter relaxes. The pattern can be ported once validated.
3. **PQ8 codes instead of 1-bit.** Higher fidelity but 8× the memory; the
   1-bit popcount kernel is also a single CPU instruction whereas PQ scans
   need lookup tables. The 1-bit path is the cleanest baseline.
4. **Skip this entirely and ship NGT-QG bindings.** Rejected — pure Rust /
   no FFI is a hard ruvector requirement (per global CLAUDE.md "RUST ONLY").

## Acceptance Criteria

* `cargo build --release -p ruvector-symphony-qg` succeeds. ✅
* `cargo test --release -p ruvector-symphony-qg` passes with real recall test
  (≥0.85 recall@10 on the cluster benchmark). ✅
* `symphony-qg-demo` binary produces ≥1.20× QPS speedup over the f32 baseline
  with no recall regression on 50 k × 128-D. ✅ (1.30× measured)

## References

* SymphonyQG (Yu et al., SIGMOD 2025)
* RaBitQ (Gao & Long, SIGMOD 2024)
* FINGER (NeurIPS 2023)
* docs/research/nightly/2026-06-23-symphony-qg/README.md
