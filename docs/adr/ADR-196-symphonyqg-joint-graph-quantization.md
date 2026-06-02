---
adr: 196
title: "SymphonyQG: joint graph + 1-bit quantization ANN crate"
status: proposed
date: 2026-06-02
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ruvector, ann, quantization, graph, rabitq, symphonyqg]
---

# ADR-196 — SymphonyQG: Joint Graph + 1-bit Quantization

## Status

**Proposed.** Companion to the nightly research at
`docs/research/nightly/2026-06-02-symphonyqg-joint-graph-quantization/README.md`.
The PoC crate `crates/ruvector-symphonyqg` is reproducible
(`cargo run --release -p ruvector-symphonyqg --bin symphonyqg-demo`).

## Context

`ruvector` already ships two RaBitQ-family crates:
`crates/ruvector-rabitq` (the original 1-bit indexer) and
`crates/ruvector-rairs` (RaBitQ-style dual-assignment IVF). Both treat
quantization as a *post-processing* layer: a separate graph or IVF index
chooses candidates with full-precision distances, and the quantizer
applies later for storage or reranking.

The SymphonyQG paper (Yang et al., SIGMOD 2025) makes the opposite
choice: the graph *itself* is traversed with quantized distances, and
the full-precision pass is reduced to a short rerank of the final
shortlist. This shaves the distance computation — the dominant cost in
graph ANN — by 20–30×, with a reported 1.5–2× end-to-end QPS at high
recall on the SIFT, DEEP, and T2I benchmarks.

`ruvector` had no joint graph-quantization indexer. This ADR proposes
adding one.

## Decision

Add `crates/ruvector-symphonyqg` as a new workspace member. It exposes
a `Searcher` over a single-layer NSW graph + bit-packed RaBitQ-style
codes, with three search modes (`Float`, `Binary`, `Symphony { rerank }`)
that share the same underlying graph.

### Scope of this ADR (PoC only)

* Single-layer NSW (not HNSW). Multi-layer routing is a follow-up.
* 1-bit codes (`sign(centred + Hadamard-rotated)`). 4-bit codes are the
  paper's production sweet spot but are deferred.
* `forbid(unsafe_code)`; scalar `u64::count_ones` kernel. SIMD popcount
  is a follow-up.
* `rand` for data generation only; the codec is deterministic
  (LCG-seeded ±1 diagonal).

### Numbers from the PoC (Apple M4 Max, rustc 1.89.0 release)

| Mode | Recall@10 | QPS | µs/query |
|---|---|---|---|
| Float graph (baseline) | 91.6% | 4,777 | 209.4 |
| Binary graph (1-bit only) | 11.5% | 10,676 | 93.7 |
| SymphonyQG (1-bit + rerank=200) | 57.9% | 8,938 | 111.9 |
| Brute force (oracle) | 100.0% | 1,934 | 517.0 |

Workload: N=8K, D=128, k=10, ef_search=200, M=24, ef_construction=128.
i.i.d. Gaussian via Box–Muller (the standard ANN-benchmark fixture).

The 32× memory compression and the 1.87× speedup at the cost of ~34 pp
recall match the expected shape of a 1-bit code on Gaussian data — the
research doc documents why and what to do next (move to 4-bit).

## Consequences

### Positive

* First crate in `ruvector` to *fuse* the graph and the quantizer.
* Clear extension path to the existing `ruvector-rabitq` codec and the
  `ruvector-hailo` cluster (4-bit popcount maps cleanly to Hailo INT4).
* Single-file demo that anyone can run to reproduce the numbers.
* All four source files under the 500-line cap.

### Negative

* 1-bit codes alone lose enough recall on D=128 Gaussian data that
  Symphony rerank cannot fully close the gap (57.9% vs 91.6% on this
  PoC). Production deployment will require 4-bit codes, documented in
  the research doc's "What to improve next" section.
* Single-layer NSW with stride-spaced multi-entry seeding is a
  PoC-grade graph; HNSW or NSG is needed for production recall.

### Risks

* **Provenance.** "SymphonyQG" is the paper's term. This crate is
  inspired by the paper, not a faithful port. The README is explicit
  about that.

## Alternatives considered

1. **Extend `ruvector-rabitq` in place.** Rejected — the existing crate
   models RaBitQ as a post-processing layer over an external graph
   builder; folding joint traversal into it would conflate two
   different abstractions (codec vs index).
2. **Wait for an HNSW upper layer.** Rejected — the PoC's
   multi-entry-seeded flat NSW is enough to demonstrate the
   speed/recall tradeoff and let downstream work focus on the
   *codec* (4-bit, learned rotation), which is where the recall
   ceiling actually sits.
3. **CAGRA-style GPU graph.** Out of scope; CPU-first PoC fits the
   existing `ruvector` workspace and the M-series target hardware.
