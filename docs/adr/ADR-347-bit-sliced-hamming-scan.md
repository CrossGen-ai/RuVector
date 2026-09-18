# ADR-347: Bit-Sliced Hamming Top-K — Layout Trade-off for Binary Embedding Vectors

## Status

Rejected (for single-query CPU workloads). Experimental crate
(`ruvector-bit-sliced-hamming`) retained as reference implementation
and as the substrate for a follow-up 64-way batched-query bit-sliced
backend (proposed but not built in this ADR). The **cache-blocked
row-major** backend from the same crate is a real, portable, 12–43%
win over the naive `NaiveRowMajor` popcount scan and is recommended as
the default binary-Hamming top-K path.

## Context

Binary embeddings — as produced by `RaBitQ` (SIGMOD 2024),
`Cohere-embed-english-v3-binary`, and 1-bit distillation of dense float
encoders — have made Hamming top-K a hot path again in the ruvector
crate family (`ruvector-rabitq`, `ruvector-fused-rabitq-residual`,
`ruvector-turboquant`). Today those crates all rely on a row-major
XOR + `u64::count_ones()` scan, which lowers to `POPCNTQ` on x86_64 and
`CNT + ADDV` on aarch64.

Recent literature (PACS ICDE 2024; bit-sliced retrieval in GPU vector
DBs like cuVS `IndexBinaryFlat` and LanceDB 0.13's GPU backend) claims
1.5–3× wins from **bit-transposed** storage combined with dim-band
early termination. Milvus 2.6 (Aug 2026) and Qdrant 1.15 (Sep 2026) both
shipped opt-in bit-sliced binary scans, but each restricts the feature
to cold-tier or batched (`k ≥ 100`) workloads. This ADR asks whether a
bit-sliced backend is worth adopting inside the ruvector workspace, and
under which conditions.

## Hypothesis

```text
Given a bit-transposed layout (dim × ceil(n/64) u64 words), a per-vector
u8 counter, and a provably lossless BAND=32 pruning sweep using
L_j = counters[j] and U_k = k-th smallest (counters[j] + remaining_dims)
as the admissibility bound, bit-sliced Hamming top-K should match or
exceed row-major POPCNT throughput on a modern out-of-order CPU for at
least one of:
  (a) random iid binary corpora, n=20k, dim=128, k=10;
  (b) 8-cluster corpora where most vectors are far from the query;
  (c) n=40k dim=64 (short row-major scans should favour bit-sliced).
```

## Method

Three backends behind one `HammingTopK` trait, cross-checked for
identical top-k on every query:

* `NaiveRowMajor` — the correctness oracle.
* `BlockedRowMajor` — cache-blocked row-major with an online max-heap.
* `BitSlicedEarlyTerm` — bit-transposed + dim-band pruning.

Benchmark binary (`cargo run --release -p ruvector-bit-sliced-hamming
--bin benchmark`) runs 500 queries × 3 regimes on this host, measures
queries/second via `std::time::Instant`, and asserts that every
non-oracle backend's top-k list is exactly equal to `NaiveRowMajor`'s
on the first 32 queries of each regime.

## Findings

Real measured numbers (aarch64 Apple Silicon, Rust 1.77, release
profile, 2026-09-18):

| regime                 | Naive q/s | Blocked q/s | BitSliced q/s | matches oracle |
|------------------------|----------:|------------:|--------------:|:--------------:|
| random n=20k dim=128   |    19 516 |      21 831 |           883 |      all       |
| clustered n=20k dim=128|    19 672 |      22 751 |         1 146 |      all       |
| random  n=40k dim=64   |    10 590 |      15 138 |           913 |      all       |

* Bit-sliced loses by 15–25× on every regime tested. Pruning on the
  clustered corpus buys back ~30% vs bit-sliced-random but is not
  enough to close the gap to row-major.
* Blocked row-major beats naive by 12–43% with no correctness change.
* Storage overhead of bit-sliced is negligible (+0.16% from
  chunk-tail padding).

## Decision

1. **Do not** enable `BitSlicedEarlyTerm` for single-query CPU
   Hamming top-K in production ruvector binary-index paths. Recommend
   `BlockedRowMajor` (or better) instead.
2. **Retain** `ruvector-bit-sliced-hamming` as a workspace member. Its
   `PackedCorpus` type, `HammingTopK` trait, and `BlockedRowMajor`
   backend are portable, dependency-light drop-in replacements for the
   inline scans currently duplicated across `ruvector-rabitq`,
   `ruvector-fused-rabitq-residual`, and `ruvector-turboquant`; a
   future consolidation ADR should migrate them.
3. **Follow-up work** (not gated by this ADR): a batched
   (`Q = 64`-query) bit-sliced backend that amortises the
   bit-transposed scan across many queries, and an
   `avx512-vpopcntdq` / SVE2 arch-specific row-major backend. See
   §"What to improve next" in the research document.

## Consequences

* Positive: We publicly commit a working, cross-backend-verified PoC
  and its benchmark numbers, which pre-empts adopting the bit-sliced
  optimisation on hearsay. The `HammingTopK` trait and the winning
  `BlockedRowMajor` backend give downstream crates a clean upgrade
  path for their current inline scans.
* Positive: Files kept small (largest is `bit_sliced.rs` at ~180
  lines), zero `unsafe`, no arch-specific intrinsics, portable across
  Apple Silicon / x86_64 / aarch64 servers.
* Neutral: We add ~1.5 kLoC of Rust to the workspace, all
  feature-independent (compiled only when `-p ruvector-bit-sliced-hamming`
  is selected or the crate is added as a dep).
* Negative: The bit-sliced backend as written is a genuine loss for the
  advertised use case. Leaving it in the tree risks a future
  contributor reaching for it without reading the ADR; the crate README
  and the `BitSlicedEarlyTerm` module docs both call this out
  explicitly.

## Alternatives considered

* **Batched bit-sliced only** — skip the single-query bit-sliced
  backend entirely and only build the 64-way batched version. Rejected
  for this iteration because we wanted a same-crate comparison against
  row-major that the trait system enforces, and because the batched
  variant is substantially more code (query bit-transpose + counter
  matrix) than a single nightly can honestly ship with real benchmarks.
* **AVX-512 `VPOPCNTDQ` row-major** — expected ~2× further speedup on
  Sapphire Rapids and later; deferred because the Apple Silicon dev
  host cannot benchmark it truthfully today.
* **Bit-serial popcount adder** replacing the `while m != 0 {
  trailing_zeros; counter++ }` inner loop — improves bit-sliced by
  another ~4× on paper but still lands well short of row-major on
  single-query CPU workloads.
* **Do nothing** — leave the inline Hamming scans in the RaBitQ and
  TurboQuant crates as-is. Rejected because `BlockedRowMajor` is a
  no-cost improvement over what those crates ship today, and the trait
  boundary makes future backend swaps trivial.

## Numeric acceptance test

The PoC either passes (bit-sliced beats blocked in at least one
regime) or the ADR must explain the gap. Bit-sliced did **not** pass;
this ADR is the explanation.

## References

* Research doc:
  `docs/research/nightly/2026-09-18-bit-sliced-hamming-scan/README.md`
* Crate: `crates/ruvector-bit-sliced-hamming/`
* Prior nightly: `docs/research/nightly/2026-09-17-hnsw-neighbor-prefetch`
* Public gist: linked from the branch push (`gh gist create` output).
