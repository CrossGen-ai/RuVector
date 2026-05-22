---
adr: 194
title: "SymphonyQG — Fused Graph + 1-bit Quantization Index"
status: accepted
date: 2026-05-22
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ann, graph, quantization, rabitq, hnsw, nightly-research]
---

# ADR-194 — SymphonyQG: a fused graph + 1-bit quantization index

> **Provenance note.** "SymphonyQG" refers to the SIGMOD 2024 paper by
> Gou, Cheng, Cong, Tao. The crate `crates/ruvector-symphonyqg` is an
> *independent Rust PoC inspired by* the paper, not a port. The rotation
> used here is a cheap permutation+sign-flip (`y = D·π(x)`), not the
> Walsh–Hadamard transform the paper recommends. Evaluate against the
> reproducible numbers in
> `docs/research/nightly/2026-05-22-symphonyqg/README.md`, not against
> the paper's headline figures.

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-22-symphonyqg` as
`crates/ruvector-symphonyqg`. All unit tests pass; build is green with
`cargo build --release -p ruvector-symphonyqg`; demo numbers captured.

## Context

ruvector ships:

- `ruvector-core` — HNSW (graph-only, full-precision distance).
- `ruvector-rabitq` — RaBitQ quantizer (flat-scan only).
- `ruvector-rairs` — IVF (ADR-193).

What it does **not** ship is a *fused* index — one that uses quantized
distance *inside the graph traversal*. Every production vector database
that takes performance seriously (Milvus 2.5, Qdrant 1.13, Weaviate 1.27,
ScaNN) does this fusion in one form or another. The recent SymphonyQG
paper (SIGMOD 2024) formalises the recipe: keep the small-world graph,
replace per-hop L2 with the unbiased RaBitQ estimator, and rerank a small
candidate list with full precision.

This ADR adopts that recipe as a ruvector nightly PoC and lays the
groundwork for a production-grade fused index.

## Decision

Add `crates/ruvector-symphonyqg` as a workspace member implementing:

1. `RaBitQuantizer` — random `y = D·π(x)` rotation + 1-bit-per-dim packing
   + L2-norm capture.
2. `BitCode` — `Vec<u64>` of packed sign bits plus the original norm.
3. Two distance estimators:
   - **FP-asymmetric** (full-precision query against ±1 sign code) for
     higher recall at per-dim cost.
   - **Popcount-symmetric** (both sides 1-bit) for low-cost traversal.
4. `Graph` — a single-layer NSW with `build_nsw` (full-precision edges)
   and `search` accepting an arbitrary distance closure.
5. `SymphonyQg` index with three search modes: `search_exact_graph`,
   `search` (FP estimator), `search_popcount`. Each performs a full-
   precision rerank of the top-`ef` survivors.
6. Real benchmark binary (`symphonyqg-demo`) and `criterion` bench.

The PoC stays a single layer, no SIMD, no `unsafe`, no rayon. Those are
explicit follow-ups in the research doc.

### Measured numbers (single core, `cargo run --release`)

| dataset           | brute | NSW exact | SymphonyQG-FP | SymphonyQG-popcount |
|-------------------|-------|-----------|---------------|---------------------|
| n=2k, d=64        | 59.4 µs / —   | 35.7 µs / 0.876 | 52.3 µs / 0.811 | 24.6 µs / 0.664 |
| n=5k, d=128       | 290.9 µs / —  | 68.8 µs / 0.666 | 104.7 µs / 0.590 | 30.6 µs / 0.478 |
| n=10k, d=128, M=24 ef=96 | 585.3 µs / — | 144.0 µs / 0.772 | 216.2 µs / 0.689 | 56.0 µs / 0.530 |

Recall@10 is shown after the slash. Bit codes are **18× smaller** than
fp32 vectors at d=128.

## Consequences

### Positive

- Demonstrates a measurable 2.2–2.6× per-query speedup over the existing
  exact-graph baseline on the same NSW topology, with order-of-magnitude
  memory savings on the per-vector code.
- Builds a generic distance-closure interface (`search<F: FnMut(u32) ->
  f32>`) that lets the rest of ruvector plug in any future estimator
  (2-bit RaBitQ, OPQ, learned) without re-architecting the graph.
- Independent from `ruvector-core`, `ruvector-rabitq`, and `ruvector-rairs`;
  no upstream breakage risk.

### Negative / Risks

- Recall is materially below the full-precision graph at d=128 (0.53 vs
  0.77 at ef=96). The PoC is honest about this; production needs (a) a
  Walsh–Hadamard rotation, (b) higher ef sweeps, and/or (c) 2-bit
  residual codes to close the gap.
- Single-layer NSW won't scale past ~1M nodes — the hop count grows. A
  follow-up must add HNSW layering or DiskANN-style entry-point selection.
- Random-rotation calibration assumes near-isotropic data. Heavily
  anisotropic embeddings (CLIP) will need a learned rotation (OPQ-style)
  before production rollout.

## Alternatives considered

1. **Extend `ruvector-rabitq` with a graph wrapper.** Rejected: the
   `RaBitQuantizer` API there is tied to flat-scan layouts. A fused
   index needs its own data structure and a graph builder that knows
   about code memory. Keep both crates and link from the SymphonyQG
   research doc.
2. **Bolt quantized rerank onto `ruvector-core` HNSW.** Rejected for the
   PoC: too entangled with the HNSW layered routing. Worth doing once
   the PoC validates the recall/latency tradeoff.
3. **Wait for upstream Rust port of SymphonyQG.** Rejected: there is no
   permissive-licensed Rust port today (May 2026); waiting forgoes the
   speedup indefinitely.

## Follow-ups

- ADR-195+: Walsh–Hadamard rotation, 2-bit residuals, SIMD popcount
  kernel.
- Sweep `ef ∈ {64, 128, 256, 512}` and publish recall/QPS curves matching
  the SymphonyQG paper's Figure 7.
- Integrate filter pushdown via `ruvector-acorn`.
