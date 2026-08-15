# ADR-306 — Residual Product Quantization (RPQ2)

## Status

Proposed — reference implementation landed as `crates/ruvector-rpq` with
measured benchmarks. Not yet integrated into the default ANN pipeline.

## Context

`ruvector` already ships several vector-compression codecs (`ruvector-rabitq`,
scalar variants inside `ruvector-turboquant`, HNSW-native f32 storage). What
we do **not** ship is a self-contained *residual* product quantizer with a
swappable trait, so that offline studies and future IVF-style indexes can pick
the best codec for a given (recall, memory, latency) tradeoff without
depending on a heavy quantizer stack.

Two-level residual product quantization ("RPQ2", closely related to the
IVFADC family of Jégou et al., 2011; the Optimized Product Quantization
family of Ge et al., 2013; and the RVQ/AQ family) is a classical but still
practically important building block. Modern high-recall ANN systems
(FAISS `IVFPQ`, Milvus `IVF_PQ`, ScaNN's asymmetric-hashing variants)
all use some form of coarse-quantize / fine-quantize decomposition.
Having a clean, dependency-free Rust implementation lets us:

1. Reproduce PQ / RPQ tradeoffs on our own datasets.
2. Provide the residual-coding primitive that a future IVFADC index needs.
3. Measure recall/latency envelopes to inform when to pick RPQ2 over
   single-level PQ, `ruvector-rabitq`, or scalar quantization.

## Decision

Add a new crate `crates/ruvector-rpq` that:

* Defines a `Quantizer` trait (encode / adc_sq_distance / code_bytes).
* Ships three interchangeable backends:
  * `Pq` — single-level product quantizer with configurable `m`, `k=256`.
  * `Rpq2` — two-level residual PQ; encoding is
    `code = [PQ1(x) || PQ2(x - PQ1⁻¹(PQ1(x)))]`.
  * `Sq8` — per-dimension 8-bit uniform scalar quantizer (baseline).
* Includes a `Rpq2Scorer` that amortises the residual LUT across all
  database entries sharing the same coarse code (the standard trick).
* Uses no external dependencies. `#![forbid(unsafe_code)]`.
* Ships a `rpq-bench` binary that runs a clean end-to-end benchmark
  (train → encode → query → recall vs. brute force).

## Consequences

**Positive.**

* Portable reference implementation of two-level RPQ we can point at in
  design docs and re-use inside future IVFADC / SPANN-style indexes.
* Trait allows plug-and-play A/B testing of quantizers.
* Zero unsafe code, zero external dependencies, deterministic (seeded).
* Real cargo-run numbers (see research doc) — no aspirational claims.

**Negative / open.**

* Our measured benchmark shows that at *equal total storage* (16 B / vec,
  dim=64), plain PQ-16 beats RPQ2(8,8) on recall@10 (**8.2 %** vs
  **5.6 %**). RPQ2's real advantage appears only when it is composed with
  a coarse layer that is amortised across many items (IVFADC), or when
  subspace dimensionality is otherwise pinned.
* The amortised `Rpq2Scorer` still requires one residual-LUT
  construction per unique coarse code. In a small database (20 k) the
  coarse codes are effectively unique, so scoring is ~400× slower than
  single-level PQ. This cost shrinks in an IVF setting where thousands
  of vectors share a coarse code.
* K-means training uses `k-means++` seeding but is single-threaded; for
  larger training sets a `rayon` parallelization would be a natural
  follow-up (kept out of this PoC to preserve the "zero-dep" property).

## Alternatives considered

1. **RVQ / additive-quantization multi-stage codebooks.** More flexible
   than RPQ2 but requires an iterative encoding search and is markedly
   harder to keep in a small dependency-free crate. Deferred.
2. **OPQ (rotated PQ).** Learns a rotation matrix to align data with
   coordinate axes before splitting into subspaces. Complementary to
   both `Pq` and `Rpq2` — planned as a follow-up crate rather than a
   feature flag here to keep code paths simple.
3. **RaBitQ.** Already in-tree via `ruvector-rabitq`; complementary,
   targets different (recall × bits) frontier.
4. **Do nothing.** Rejected — we need at least one dependency-free
   reference PQ implementation to unblock future work and to sanity-check
   external codec claims.

## Files touched

* `crates/ruvector-rpq/Cargo.toml`
* `crates/ruvector-rpq/src/lib.rs`
* `crates/ruvector-rpq/src/bin/bench.rs`
* `Cargo.toml` (workspace membership)
* `docs/research/nightly/2026-08-15-residual-product-quantization/README.md`
* `docs/research/nightly/2026-08-15-residual-product-quantization/gist.md`
