# ADR-356: Hamming-Weight Pre-Bucketing for Binary ANN

**Status:** Proposed
**Date:** 2026-10-20
**Deciders:** nightly research (crossgen-ai)
**Related:** `gist-bit-sliced-hamming-2026`, `gist-partial-l2-early-abandon-2026`, `gist-pq-adc-early-abandon-2026`

## Context

ruvector's binary-code backends (SimHash, 1-bit RaBitQ, ternary, 1-bit scalar
quantization) all end their query path in the same inner loop: XOR + popcount
across the full code table. The crates that already live in the tree
(`ruvector-bit-sliced-hamming`, `ruvector-rabitq`, `ruvector-ternary-...`)
optimize *that inner loop*. None of them prune *which codes reach it*.

For Hamming distance, a trivial but tight lower bound exists:
`H(x, q) >= |pop(x) - pop(q)|`. The quantity `pop(x)` is an integer in
`[0..D]`, so bucketing codes by it is a `(D+1)`-way static partition that
costs ~25 % memory and takes O(N) to build. At query time, iterating
buckets outward from `pop(q)` and breaking when the delta exceeds the
current k-th-best Hamming τ gives an **exact** k-NN with provably fewer
candidate visits.

Nothing in the ruvector tree or in Milvus / Qdrant / Weaviate / LanceDB
/ FAISS exposes this bound as an index layer. The signature-file
literature uses the related Jaccard-weight bound for set-similarity joins,
but not for vector search.

## Decision

Introduce `ruvector-hamming-weight-prebucket`, a standalone crate that
exposes a `BinaryScanner` trait and three backends:

* `FullScan` — baseline, included for benchmarking.
* `WeightedScan` — the new index: `[Vec<u32>; D+1]` of ids grouped by
  popcount, iterated outward from `pop(q)` with triangle-bound pruning.
* `WeightedBitSlice` — same index with a flat `Vec<u64>` code layout for
  a tighter inner loop, demonstrating that the pre-bucket layer composes
  with any downstream scanner (SIMD, bit-sliced, PQ-ADC, ...).

The crate is deliberately dependency-free (`#![forbid(unsafe_code)]`,
only `std`) so it can be dropped into any existing binary backend. Code
width is a `const generic` so `[u64; 2]` through `[u64; 8]` compile
without allocations.

## Consequences

### Positive

* **Exact k-NN** (recall@k = 1.0 against full scan, verified on 128
  randomized queries). The pruning is a mathematical bound, not a
  heuristic — unlike quantized reranking, there is no recall knob.
* **2.13× end-to-end speedup, 3.14× visit prune** on class-structured
  codes (N = 100 000, D = 128, k = 10) — the regime that matches
  learned-to-hash and mode-separated embeddings.
* **Trivial to retrofit**: any existing binary scanner implements
  `BinaryScanner` by delegating to its current `knn`; the pre-bucket
  layer wraps around it. No API break.
* **Memory overhead ≈ 25 %** of code storage, independent of D for
  large N.
* **Build cost is O(N·W)**, same complexity as copying the codes once.

### Negative

* **Zero or negative win on symmetric SimHash** (popcount concentrated
  at D/2). Measured 0.70× on unbiased SimHash. Mitigation: a build-time
  check on `stddev(pop)` can gate the index automatically.
* **Static index**: streaming inserts are cheap (push), deletes require
  tombstoning or swap-remove. Not addressed in this iteration — the
  streaming variant is explicitly scoped out and left for future work
  alongside `ruvector-streaming-qng`.
* **Hamming-specific**: the triangle bound uses the hypercube metric.
  Jaccard and bit-cosine admit weaker weight-band bounds; integer-code
  metrics (ternary) need a different inequality. The trait signature
  is `const W` and `Code = [u64; W]`, so metric generality is a future
  trait extension, not a redesign.

### Alternatives considered

1. **Multi-index hashing (Norouzi 2012).** Partitions code bits, not codes,
   and radius-prunes per sub-bucket. Composable with weight pre-bucketing
   but much more code and depends on radius queries, not top-k.
2. **IVF on binary codes.** Builds a learned coarse partition. Costs
   training, loses exactness, and does not use the (free, provable)
   weight bound.
3. **Sorted-by-popcount layout only (no buckets).** Would scan linearly
   in weight order with the same bound, but random access patterns in
   `Vec<Vec<u32>>` are better for the current scalar scanner. Keep as
   a possible layout variant.
4. **Precomputed Hamming prefix sums per weight bucket.** Overkill for
   128-bit codes; useful only for very wide codes.

## Implementation reference

* Crate: `crates/ruvector-hamming-weight-prebucket/`
* Research: `docs/research/nightly/2026-10-20-hamming-weight-prebucket/README.md`
* Reproduce numbers:
  ```
  cargo test   --release -p ruvector-hamming-weight-prebucket --lib
  cargo run    --release -p ruvector-hamming-weight-prebucket --bin benchmark
  ```
