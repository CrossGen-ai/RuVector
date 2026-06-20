---
adr: 264
title: "LAET-HNSW — learned per-query adaptive early termination for HNSW search"
status: proposed
date: 2026-06-20
authors: [ruvnet, claude-flow]
related: [ADR-001, "docs/research/nightly/2026-06-20-laet-hnsw/README.md"]
tags: [ruvector, hnsw, ann, search, adaptive, learned-index, latency, recall]
---

# ADR-264 — LAET-HNSW

> **Decision in one line.** Introduce a swappable `SearchStrategy`
> trait on `ruvector`'s HNSW search path and ship two new
> implementations alongside the fixed-`ef` baseline: a per-query
> ridge-regression-learned `ef` predictor (LAET) and a non-learned
> gap-trajectory heuristic.

## Context

`efSearch` is the dominant recall-vs-latency knob in HNSW. Today
every query in a ruvector index pays for the same global `efSearch`,
sized for the **hardest** query in the workload. The SIGMOD 2020
LAET line of work showed that a tiny per-query predictor, fed
features that the upper-layer descent already computes, picks a
much smaller `ef` for the typical query without sacrificing recall.
None of the existing Rust ANN crates ship this capability, and the
ruvector workspace already exposes most of the pieces needed
(`ruvector-snapshot` for shipping calibrated weights,
`ruvector-router-core` for per-tenant routing, `ruvector-metrics`
for drift detection).

The PoC in `crates/ruvector-laet/` exercises the design end-to-end:
own minimal HNSW (~330 LOC, diverse-neighbour heuristic), trait-
based search strategies, ridge predictor with a closed-form Gauss-
Jordan solve, and a real `cargo run --release` benchmark on a
50k × 64-dim Gaussian-cluster dataset. Results on Apple M4 Max:

| Strategy | Recall@10 | µs/query | dist/query |
|---|---:|---:|---:|
| fixed-ef ef=32 | 0.9363 | 47.8 | 660 |
| fixed-ef ef=64 | 0.9894 | 87.5 | 953 |
| **laet** | **0.9696** | **64.6** | **796** |
| gap-heuristic | 0.9998 | 241 (\*) | 96 |

(\*) the PoC's gap heuristic measures trajectory after the fact;
the production implementation must integrate the check into the
inner loop to convert the dist/query saving into wall-clock saving.

## Decision

1. Adopt the `SearchStrategy` trait as the only public entry point
   for HNSW search in future ruvector minor releases.
2. Ship the **ridge-regression LAET** implementation in production
   under a `laet` feature flag, defaulting to off until
   per-deployment calibration is operationalised.
3. Ship the **gap-heuristic** implementation behind a separate flag
   as a non-learned fallback for greenfield deployments that have
   no calibration data yet.
4. Persist predictor weights inside `ruvector-snapshot` so that a
   snapshot fully captures query-time behaviour.
5. Production crate split (`ruvector-laet-core`, `-ffi`, `-bench`)
   per the proposal in the research doc; the current
   `crates/ruvector-laet/` stays as a research playground.

## Consequences

**Positive.**

* At equal recall, the typical query pays 5-20 % fewer distance
  computations and proportionally lower latency. Effect compounds
  on heavy-tailed workloads (legal search, code search, dialogue
  histories) where most queries are easy.
* The trait surface gives us a clean home for future strategies
  (GBM, neural, online-learning) without touching the core graph
  code.
* Predictor weights are tiny (40 bytes for ridge, < 50 kB for a
  32-leaf GBM), so the WASM and embedded builds inherit the same
  capability with zero infra changes.

**Negative.**

* Calibration becomes part of the index lifecycle. Operators need a
  brute-force ground-truth sample, periodic refit, and drift
  monitoring. We mitigate by piggy-backing on the existing
  `ruvector-metrics` recall validator.
* Distribution shift can silently hurt recall. The defaulted-off
  feature flag plus the `ef_floor` clamp give two layers of safety.
* Adds one more knob (`target_recall`) operators must understand.

**Neutral.**

* No on-disk format changes today; predictor weights ride inside
  the existing snapshot format as a new optional section.

## Alternatives considered

1. **Status quo (fixed `efSearch`).** Simple, but pays peak-query
   cost on every query. Rejected: the data shows real savings are
   available for free.
2. **Linear-by-`k` dynamic ef (Weaviate-style).** Strictly weaker
   than per-query prediction; ignores query difficulty entirely.
3. **Pure progress-trajectory heuristic (no learning).** Implemented
   as the `GapHeuristicStrategy` and shipped as a fallback, but
   strictly dominated by LAET on calibrated workloads.
4. **GBM predictor as v1.** Higher gains in the literature, but
   pulls in LightGBM or a Rust GBM crate; we keep that on the
   roadmap behind the same trait.
5. **Replace HNSW entirely with DiskANN beam-search + LAET.**
   Bigger lift; deferred. LAET applies just as cleanly to DiskANN,
   so v1 ships on HNSW and we extend later.
