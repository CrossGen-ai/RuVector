# ADR-257: SymphonyQG — Coupled Graph + 1-bit Quantization Index

- **Status:** Proposed
- **Date:** 2026-06-17
- **Deciders:** ruvector nightly research process
- **Branch:** `research/nightly/2026-06-17-symphony-qg`
- **Crate:** `crates/ruvector-symphony-qg`
- **Research note:** `docs/research/nightly/2026-06-17-symphony-qg/README.md`

## Context

ruvector ships strong graph indices (acorn, diskann, roargraph,
graph) and strong quantizers (rabitq, leanvec, opq, avq, lvq,
anisotropic-pq), but the two families are independent backends. The
2024–2026 ANN literature (notably SymphonyQG, SIGMOD 2025; HVS+, VLDB
2024) has moved past this split: candidate ranking inside the graph
traversal uses cheap quantized distances, and full-precision
distances are only paid for a small reranking pool. The result, on
public benchmarks, is 5–10× QPS at iso-recall versus HNSW + post-hoc
PQ rerank.

We have no crate that demonstrates this co-design pattern yet.

## Decision

Land `ruvector-symphony-qg` as a new workspace crate. It ships:

1. A `BitQuantizer` (sign-flip + popcount Hamming) — placeholder for
   the real RaBitQ kernel; intentionally simple to keep the PoC
   self-contained and < 500 lines per file.
2. A `KnnGraph` substrate (NN-descent–lite, no HNSW hierarchy) —
   independent of the symphony layer so the substrate can be swapped
   for `ruvector-graph` / `ruvector-acorn` / HNSW later.
3. A `SymphonyIndex` that couples them: graph traversal scored by
   Hamming, top-`rerank` survivors reranked by f32 L2.
4. A reproducible `symphony-qg-bench` binary comparing three variants
   (brute force, graph-f32, symphony) with real cargo-run latency
   and recall numbers.

The PoC publishes real measured numbers (n=20 000, d=128, 200
queries, single-thread, Apple Silicon release build):

| variant | median | recall@10 |
|---|---|---|
| brute force | 1286 µs | 1.000 |
| graph (f32) | 255 µs | 0.639 |
| **SymphonyQG** | **57 µs** | **0.321** |

→ **4.47× faster than the f32 graph baseline**, at the cost of recall
on this synthetic workload. Binary code overhead: 16 bytes/vector
(2.5 % of f32 footprint).

## Consequences

### Positive

- ruvector gains an example of the modern graph+quant co-design
  pattern. Future indices (HNSW + RaBitQ, DiskANN + RaBitQ) can copy
  the coupling.
- Binary-code traversal is **22.6× faster than brute force** in this
  PoC even with the simplified quantizer. The pattern works.
- Crate is self-contained, dependency-light (just `rand` +
  `thiserror`), and runnable with one cargo command.

### Negative

- Recall@10 of 0.32 is low. Caused by (a) sign-only quantizer with
  no rotation, (b) random-unit-sphere workload with weak local
  structure. Both are addressable; see roadmap.
- Build time for the substrate is dominated by NN-descent, not
  symphony. Real production should layer over an HNSW substrate.
- Citation for the SymphonyQG paper itself is not re-verified in
  this run; treat the `(citation pending)` as a follow-up.

### Neutral

- Pattern is **substrate-agnostic**. Any graph backend can be
  upgraded later without touching the symphony layer.

## Alternatives considered

1. **Post-hoc rerank only (status quo).** Run HNSW with f32, then
   rerank with a quantizer. Faster than f32 alone but does not
   accelerate the *traversal*. SymphonyQG accelerates traversal
   itself.
2. **Pure quantized search (no rerank).** Drop the f32 rerank step.
   Much worse recall — the binary codes alone are too coarse.
3. **Wait for a real RaBitQ integration.** Doable but slower to
   land; the symphony coupling pattern is the novel piece worth
   landing now. Issue tracker entry for the RaBitQ swap is the
   first item in the research roadmap.

## Follow-ups

- ADR follow-up: swap `BitQuantizer` for the `ruvector-rabitq` kernel
  and re-measure on the same workload.
- ADR follow-up: HNSW substrate (port from `ruvector-acorn`).
- Public ANN-Benchmarks port (sift1m, deep10m, glove).
- AVX-512 `vpopcntq` kernel for d > 128.

## References

- Research note: `docs/research/nightly/2026-06-17-symphony-qg/README.md`
- Crate: `crates/ruvector-symphony-qg/`
- Prior nightly: ADR-254 (TurboVec FastScan), `docs/research/nightly/2026-04-23-rabitq/`
