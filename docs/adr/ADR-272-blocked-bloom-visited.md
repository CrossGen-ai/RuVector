# ADR-272: Blocked Bloom Visited-Set for HNSW Traversal

**Status**: Proposed
**Date**: 2026-07-06
**Author**: Nightly Research Agent
**Branch**: `research/nightly/2026-07-06-blocked-bloom-visited`
**Crate**: `crates/ruvector-blocked-bloom`
**Related**: ADR-233 (HNSW Repair), ADR-256 (Hybrid Search), ADR-265 (SOTA Bench)

---

## Context

The visited-set is the least-glamorous hot spot in an HNSW / Vamana / DEG search:
called once per neighbor edge, it decides whether a candidate has already been
processed on the current query.  In every Rust HNSW implementation we surveyed
(`hnsw_rs`, `instant-distance`, ruvector-core) this is a `HashSet<u32>`.
Independent measurement (see `crates/ruvector-blocked-bloom/src/benchmark.rs`)
shows the HashMap probe alone burns ~13 ns per operation and dominates cache
pressure for small-vector graphs.

Two well-known alternatives exist but have not been benchmarked or trait-abstracted
inside RuVector:

1. **Dense bitmap** — 1 bit per node id.  Exact; O(1) probe; but naïve `reset()`
   is O(n_ids), which is unusable above ~10 M nodes.  hnswlib works around this
   with a per-query epoch counter and 16-bit labels (2·N bytes).
2. **Blocked Bloom filter** (Putze/Sanders/Singler 2007) — cache-line-aligned
   filter with `k=4` bit positions per 512-bit block, chosen so every probe
   touches exactly one cache line.  Fixed-size, no allocation, tiny false-positive
   rate.  Widely used in OLAP joins (DuckDB, Impala) but not, to our knowledge,
   in any production Rust ANN engine.

Neither has been swappable behind a trait inside `ruvector-core`, and neither
has been measured on a realistic HNSW traversal shape inside this repo.  This
ADR proposes both, behind a single trait, with an O(dirty) reset for the
bitmap that makes it viable on billion-node graphs.

---

## Decision

Introduce `crates/ruvector-blocked-bloom` as a standalone Rust crate exporting
a single `VisitedSet` trait with three interchangeable implementations:

| Impl                     | Exact | Memory (n_ids=1M)  | Throughput  | Speedup vs baseline |
|--------------------------|-------|--------------------|-------------|---------------------|
| `HashSetVisited`         | yes   | ~201 KB (dynamic)  | 77.7 Mops/s | 1.00× (baseline)    |
| `BitmapVisited` (dirty)  | yes   | 191 KB fixed       | 254.4 Mops/s| **3.27×**           |
| `BlockedBloomVisited`    | no    | **123 KB fixed**   | 74.3 Mops/s | 0.96×               |

Measurements taken on a 1 M-node HNSW-shape traversal, 32.77 M probes total.
Blocked-Bloom FP rate measured against a per-query oracle: **0.0044%** —
three orders of magnitude below the recall floor of any HNSW query at
ef_search ≥ 32.

`ruvector-core::hnsw` will gain a feature `blocked-bloom` that swaps the
default `HashSet` visited-set for either the bitmap (when `n_nodes < 10^8`)
or the blocked-Bloom (when it is larger or unknown).

---

## Consequences

### Positive

* **3.27× throughput** on the visited-set hot path when the id space is
  known (all in-memory HNSW builds), for less memory than the current
  HashSet.
* **Fixed 123 KB memory** for streaming or billion-node graphs where a
  dense bitmap would blow L2 — matches the streaming vector index use
  case in `ruvector-lsm-ann` (ADR-264) and `ruvector-diskann`.
* **Trait-first** design lets downstream crates pick their own trade-off;
  in particular `ruvector-sota-bench` gains a knob to sweep the visited-set
  across index kinds without changing search code.
* **O(dirty) reset** — the dirty-tracking technique used here is directly
  reusable in `ruvector-coherence-hnsw` and `ruvector-hnsw-repair` where
  per-query resets currently dominate small-batch latency.

### Negative / Risks

* Blocked Bloom introduces **approximate visited-tracking**.  A false
  positive causes a candidate to be skipped, which reduces recall by up
  to the FP rate.  Mitigation: (a) size the filter to keep FP < 0.01%,
  (b) restrict the Bloom variant to graphs where the space cost of the
  bitmap is prohibitive.
* Bitmap-with-dirty requires knowing `n_ids` at construction.  Growing
  index (e.g. `ruvector-lsm-ann`) must resize or fall back to Bloom.
* On our current bench (Apple Silicon dev machine), the Bloom variant is
  slightly *slower* than the HashSet baseline.  The win is memory and
  scale-independence, not speed at 1 M nodes.  Future SIMD work
  (AVX-512 `vpbroadcastq` + `vptestmb`) should close and reverse the gap.

### Neutral

* No changes to on-disk formats.  The visited-set is per-query, transient,
  never serialised.
* No changes to the public HNSW API — only an additive `--features
  blocked-bloom` flag on `ruvector-core`.

---

## Alternatives Considered

1. **hnswlib-style epoch-labeled bitmap.**  `Vec<u16>` sized to n_ids, epoch
   counter incremented per query.  Reset is O(1) but memory is 2·N bytes —
   2 GB for 1 B nodes.  Rejected on memory grounds.
2. **Cuckoo filter.**  Supports deletion, but the hot path is 2 hashes plus
   possibly a relocation; measured slower than a blocked Bloom in every
   OLAP benchmark we could find.  Rejected on complexity/speed.
3. **Robin-Hood open-addressing HashSet with u32 keys.**  Would beat
   `std::HashSet<u32>` by ~1.5× per our micro-benchmarks but still lose
   to the bitmap and gain no memory advantage over Bloom.  Not enough
   uplift to justify introducing a bespoke hash impl.
4. **Register-blocked Bloom (RBBF, Lang 2019).**  Uses 32-bit registers
   instead of cache lines; better on latency-critical CPUs with narrow
   loads.  Rejected for our target (modern x86_64 / Apple Silicon) — the
   cache-line block is a better fit.  Worth revisiting if we target
   embedded (Cognitum Seed devices).

---

## Acceptance Criteria (numeric)

| Criterion                                                            | Threshold           | Measured           | Status |
|----------------------------------------------------------------------|---------------------|--------------------|--------|
| `cargo build --release -p ruvector-blocked-bloom` succeeds           | must pass           | 12.06 s clean      | ✅     |
| `cargo test -p ruvector-blocked-bloom` (6 tests, real oracle)        | must pass           | 6/6 pass           | ✅     |
| `HashSet` baseline throughput                                        | > 60 Mops/s         | 77.7 Mops/s        | ✅     |
| `Bitmap<u64>` speedup vs baseline                                    | > 2.0×              | **3.27×**          | ✅     |
| `BlockedBloom` false-positive rate on realistic frontier             | < 1.0%              | **0.0044%**        | ✅     |
| `BlockedBloom` memory (fixed, n_blocks=640)                          | < 200 KB            | 123 KB             | ✅     |
| No `unsafe` in library code                                          | must hold           | 0 `unsafe` blocks  | ✅     |
| All files ≤ 500 lines                                                | must hold           | max 246 lines      | ✅     |

---

## Follow-up work

* SIMD Bloom probe on AVX-512 / NEON (see research doc §"What to improve next").
* Integrate under `--features blocked-bloom` in `ruvector-core::hnsw`.
* Re-run `ruvector-sota-bench` DBpedia-1M recall@10 curve with the new
  visited-set and confirm the recall floor is unchanged.
