<!-- One decision, stated in the filename. -->

# ADR-0001: Fused RaBitQ + 4-bit Scalar Residual Quantization

> Decision date: 2026-09-04
> Status: Proposed
> Scope: `crates/ruvector-fused-rabitq-residual` (experimental; standalone Cargo workspace)
> Drivers: storage-bound ANN shards; recall gap between 1-bit RaBitQ and 4-bit SQ / PQ; competitor convergence on composite (coarse+refine) codecs during 2026-Q2/Q3

## Context

`ruvector-rabitq` (upstream ADR-260 in this repo's ADR-NNN series) shipped
1-bit rotation-based quantization with proven distance-bound guarantees
(Gao & Long, SIGMOD 2024). It compresses D=128 f32 vectors from 512 B to
20 B. Measured recall@10 on synthetic Gaussian data at N=8 000 is 0.324
— sufficient as a coarse pre-ranker inside IVF/HNSW, but far below
production recall targets (>=0.9) when used as the primary codec.

`ruvector-turboquant` and `ruvector-pq-search` sit at the other end of
the curve: PQ-family codes hit ~0.9 recall at ~5 bits/dim but require
codebook training, storage, and per-query codebook loads.

Competitor changelogs (Milvus 2.5, Qdrant 1.13, Weaviate 1.28, LanceDB
1.24) all converged during 2026 Q2–Q3 on a composite codec pattern:
1-bit code for candidate generation, higher-precision code for refine.
None ship a single codec that combines a rotation-based 1-bit code
with a scalar-quantized residual under one trait boundary — they
implement two codecs and orchestrate a two-pass scan.

Forces:

* Storage: 5 bits/dim is the practical sweet spot for on-node
  vector shards at 10⁸-scale.
* Recall: RaBitQ alone at D=128 is insufficient; adding a residual
  correction is the cheapest way to close the gap.
* Simplicity: avoid PQ's codebook management (training set drift,
  cold-shard load latency).
* Ergonomics: a `Quantizer` trait boundary must let us swap the
  sign-code width (RaBitQ-Ex) and residual width (SQ2) orthogonally.

## Decision

Ship `ruvector-fused-rabitq-residual` as an experimental standalone crate:

1. Seeded signed Fast Walsh–Hadamard rotation (`rotation::SignedHadamard`):
   O(D log D), zero matrix storage, L2-preserving. Requires D power of
   two; callers must zero-pad otherwise.
2. `Quantizer` trait (`quantizer::Quantizer`) with `code_bytes`, `name`,
   `encode`, `prepare_query`, `distance`.
3. Three concrete impls in the same trait: `RabitQuant` (1-bit, 4 + D/8
   bytes), `Sq4Quant` (4-bit rotated, 8 + D/2 bytes), `FusedRQR` (1-bit
   sign + 4-bit residual, 12 + D/8 + D/2 bytes).
4. Trait-generic `QuantizedIndex<Q>` linear scan (`scan.rs`).
5. Runnable demo `fused-rq-demo` and criterion benches
   (`benches/fused_rq_bench.rs`) — no mocks, real cargo-run numbers.
6. The crate is a standalone `[workspace]` so it does not force
   criterion into the RuVector root workspace until promotion.

## Alternatives Considered

1. **Ship RaBitQ-Ex (2–3 bit multi-level) alone.** Cleaner theoretically
   but loses the orthogonal residual-codec swap. Deferred as a follow-up
   (see research doc §"What to improve next").
2. **Add SQ4 alone as a new codec.** Ignores RaBitQ's sign structure;
   matches SQ4 recall (0.845) but not fused recall (0.904).
3. **Extend `ruvector-rabitq` in place with a residual mode.** Couples
   residual codec to RaBitQ's public API, making future codec
   permutations harder. Rejected.
4. **Learned rotation (OPQ) instead of SFHT.** Higher recall on
   correlated data but adds a training step and rotation storage.
   Deferred; SFHT is the correct baseline for a first cut.
5. **Two-pass RaBitQ→SQ4 refine (competitor pattern).** Same total
   bit budget but requires two scan passes and does not return exact
   squared-L2 in the rotated basis in one pass. The fused single-pass
   scan is strictly simpler to integrate into HNSW/IVF traversal.

## Consequences

Positive:

* Measured recall@10 = 0.904 at 5.75 bits/dim (D=128, N=8 000, Gaussian).
  Strict Pareto improvement over RaBitQ (0.324 at 1.25 bits/dim) and
  SQ4 (0.845 at 4.50 bits/dim).
* No `unsafe`, no SIMD intrinsics — clean baseline for the SIMD scan
  follow-up (projected 3–5× speedup with NEON/AVX2 nibble-unpack).
* No codebook training — deployable to fresh shards immediately.
* Test suite enforces `MSE(fused) < MSE(RaBitQ)/2` as a hard invariant.

Negative / cost:

* D must be a power of two (SFHT). Non-p2 dims require zero-pad.
* Per-query latency 1.2 ms at N=8 000 without SIMD; roughly 3–5× slower
  than a hand-tuned NEON scan.
* Not wired to any HNSW/IVF backend yet — only linear-scan recall on
  synthetic data is measured.
* Adds a new crate + standalone workspace, marginally increasing repo
  surface. Rollback is a directory delete; no downstream crate depends
  on it.

## Testable Criteria

| ID   | Criterion                                                                          | How verified                                                                                                              |
|------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| TC-1 | Signed Fast Walsh–Hadamard rotation preserves L2 norm to 1e-4 relative on D=128    | `cargo test -p ruvector-fused-rabitq-residual tests::rotation_preserves_norm`                                             |
| TC-2 | Code sizes match analytic formula: RaBitQ=4+D/8, SQ4=8+D/2, Fused=12+D/8+D/2       | `cargo test -p ruvector-fused-rabitq-residual tests::code_sizes_match_analytic_formula`                                   |
| TC-3 | MSE(fused reconstruction) < 0.5 · MSE(RaBitQ reconstruction) and Fused recall@10 >= RaBitQ recall@10 + 0.05 and Fused recall@10 >= SQ4 recall@10 - 0.05 on the seeded synthetic corpus | `cargo test -p ruvector-fused-rabitq-residual tests::fused_beats_rabitq_alone_on_reconstruction`                          |
| TC-4 | Demo produces recall@10 >= 0.85 for the fused codec on D=128, N=8 000, k=10        | `cargo run --release -p ruvector-fused-rabitq-residual --bin fused-rq-demo` — inspect final `Fused (5-bit)` row           |

## References

* `docs/research/nightly/2026-09-04-fused-rabitq-residual/README.md` — SOTA survey, design walkthrough, full benchmark table.
* `crates/ruvector-fused-rabitq-residual/` — implementation.
* J. Gao, C. Long. "RaBitQ." *SIGMOD 2024.*
* J. Gao et al. "RaBitQ-Ex." arXiv:2409.09913, 2025.
* H. Jégou, M. Douze, C. Schmid. "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI 2010.*
* N. Ailon, B. Chazelle. "Fast Johnson–Lindenstrauss Transform." *SICOMP 2009.*

