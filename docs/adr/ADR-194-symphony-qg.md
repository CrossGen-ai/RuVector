# ADR-194 — Symphony-QG: graph + quantization fusion for ruvector

* Status: Proposed (PoC landed under `crates/ruvector-symphony-qg`)
* Date: 2026-05-17
* Author: nightly research agent
* Related ADRs: ADR-193 (RaIRS IVF), ADR concerning ruvector-rabitq

## Context

Most production ANN stacks today still follow the "PQ + rerank" pattern:
scan all (or many) product-quantized codes with an Asymmetric Distance
Table (ADT), keep the top-`r` candidates, then rescore those with full
precision. Reranking is correct but expensive: it adds a random-access
load of `r` raw vectors per query, which is the dominant bottleneck
when vectors live in slow memory (HBM, NVMe, S3).

SymphonyQG (Gou et al., SIGMOD 2025) proposes a tighter coupling: build
a navigable small-world graph whose traversal uses the *same quantized
metric the user will pay for at query time*. If the graph is good
enough and the quantizer is precise enough, the rerank pass can be
elided entirely — the headline savings are read bandwidth, not just
distance ops.

ruvector already ships RaBitQ, AVQ, LeanVec, RaIRS, MUVERA, and a flat
HNSW. We do not have a single crate that demonstrates the
graph + PQ fusion as a swappable backend, nor measured numbers
quantifying how close ADT-only graph search can get to the PQ+rerank
recall ceiling on our synthetic harness.

## Decision

Add a new crate `ruvector-symphony-qg` that:

1. Defines an `AnnIndex` trait with three reference implementations:
   * `FlatIndex` — brute-force f32 baseline (ground truth).
   * `PqRerankIndex` — full ADT scan + top-`r` full-precision rerank.
   * `SymphonyQgIndex` — alpha-RNG-diversified small-world graph
     built on full-precision distances; *searched* via ADT scoring
     and farthest-point-sampled multi-entry traversal.
2. Exposes an optional `with_refine(r)` knob that rescores the top-`r`
   graph candidates with full precision. This is the explicit
   "recall vs. latency" lever the index gives operators.
3. Ships a `symphony-qg-demo` binary, real `cargo test` recall floors,
   and a `cargo bench` harness (`criterion`).

## Consequences

### Positive

* Establishes a clean abstraction (`AnnIndex`) inside the crate so
  future quantizers (RaBitQ-graph, BBQ-graph) can plug into the same
  benchmark harness.
* The PoC produces real, reproducible numbers that show *exactly*
  where Symphony-QG wins and loses (see research doc).
* Memory math is documented and verified by code (`compression_ratio`).

### Negative / known limitations

* On hard configs (large `n`, high `d`, or low `M`), ADT-only graph
  recall falls well below the PQ+rerank baseline. Measured: at
  `n=20k, d=64, M=16` Symphony recall@10 is 13 % vs. PQ+rerank 98 %.
  This matches the theoretical expectation that ADT quantization
  noise compounds along long traversal paths. The crate exposes
  `with_refine` to recover most of the gap at small latency cost.
* Build is O(n²) in the PoC. The upstream paper uses a hierarchical
  build similar to HNSW; that is left for a follow-up.
* Graph stores `m_edges * 4` bytes per node — same envelope as HNSW.

### Alternatives considered

* **Pure HNSW + PQ post-pass (status quo).** Simpler, but pays the
  full rerank load every query.
* **RaBitQ-graph.** Could be a follow-up; RaBitQ has theoretical
  error bounds that may interact better with graph traversal than
  PQ does. Symphony-QG's contribution (the fusion pattern) is
  orthogonal to the quantizer choice.
* **Disk-resident SPANN-style hybrid.** Different problem class
  (memory-disk hierarchy); not in scope.

## Acceptance criteria (met by this commit)

* `cargo build --release -p ruvector-symphony-qg` succeeds.
* `cargo test -p ruvector-symphony-qg` passes 4 tests, including a
  recall floor for Symphony-QG headline (≥ 0.50) and a strict
  monotonicity test for `with_refine`.
* `symphony-qg-demo` produces real numbers reproduced in the research
  doc; no mocks, no placeholders.

## Follow-ups

* Hierarchical (HNSW-style) build to remove the O(n²) wall.
* Replace alpha-RNG with the upstream "quantization-aware diversifier"
  that incorporates ADT noise variance into the dominance check.
* RaBitQ-graph variant under the same `AnnIndex` trait.
