---
adr: 194
title: "SeRF — Segment-Graph Range-Filtered ANN"
status: accepted
date: 2026-05-25
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [ann, vector-search, range-filter, segment-tree, hybrid-query, nightly-research]
---

# ADR-194 — SeRF: Segment-Graph Range-Filtered ANN for ruvector

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-25-serf-range-filter-ann`
as `crates/ruvector-serf`. Build is green with
`cargo build --release -p ruvector-serf`; all four correctness tests pass with
`cargo test --release -p ruvector-serf`; `cargo run --release -p ruvector-serf
--example range_bench` produces the numbers cited below.

## Context

ruvector has no first-class story for *range-filtered* approximate
nearest-neighbor search — "k-NN among items with key in [lo, hi]". This is
the hybrid-query workload behind real production traffic: time-bounded RAG,
price-range product search, score-window recommender candidates. The two
classical options are:

* **Pre-filter brute-force** — fine for selective ranges, linear in subset
  size for broad ones.
* **Post-filter on a single global graph** — fine for broad ranges, recall
  collapses on selective ones because the graph doesn't know about the
  predicate. We measure 0.085 recall@10 at 1% selectivity on a 20k-vector
  workload (see `crates/ruvector-serf/examples/range_bench.rs`).

Recent SOTA — SeRF (VLDB 2024, Zuo & Deng) and iRangeGraph (SIGMOD 2024,
Xu et al.) — solve this with a segment tree over rank-space, with one graph
per canonical node. Nothing comparable exists in ruvector.

## Decision

Add `crates/ruvector-serf` providing three pluggable range-filtered ANN
backends behind a shared `RangeAnn` trait:

1. `flat::Flat` — brute-force ground truth.
2. `nsw_post::NswPost` — single global NSW with postfilter + configurable
   overscan; the classical "graph+postfilter" baseline.
3. `segment::SegmentGraph` — segment tree of NSW graphs over rank-space
   (the iRangeGraph approach; SeRF's edge-interval compression is left as a
   tracked follow-up).

The crate has **zero external dependencies** (including dev-deps); the NSW
graph is hand-rolled (~150 LoC) with deterministic insertion. Distance is
squared L2 in v0.1; the trait surface is small enough that adding IP/cosine
is one extension.

## Consequences

### Measured (real numbers, N=20 000, D=128, k=10, NQ=200, Apple Silicon, release build)

| selectivity | flat (truth)         | nsw-postfilter        | serf-segment-graph         |
|------------:|----------------------|-----------------------|----------------------------|
|        100% | 1.000 / 718 µs       | 0.517 / 147 µs        | **0.748 / 419 µs**         |
|         10% | 1.000 / 100 µs       | 0.410 / 137 µs        | **0.859 / 158 µs**         |
|          1% | 1.000 / 17.5 µs      | 0.085 / 155 µs        | **1.000 /  8.5 µs**        |

Build times: flat 1.1 ms; nsw-post 2.1 s; segment-graph 10.5 s (160 sub-graphs
constructed, 28.6 MB adjacency total vs 3.3 MB for the single-graph baseline).

### Positive

* First range-filtered ANN in ruvector with publishable numbers.
* Trait-based design lets us bolt on `serf-compressed`, `acorn-range`, or
  filtered-DiskANN backends without changing the benchmark harness.
* Segment-graph is **simultaneously faster and more accurate** than the
  post-filter baseline at narrow selectivities — exactly the regime where
  post-filter is known to fail.

### Negative / known limitations

* **Memory.** Each item lives in O(log(n/leaf_size)) graphs ≈ 6× at our
  default. SeRF's compressed form would close this gap; tracked as roadmap
  item 1 in the research README.
* **Inserts.** Every insert rebuilds O(log n) graphs. Effectively batch-only
  until the compressed/dynamic variant lands.
* **L2 only in v0.1.** Trait-extension follow-up.
* **Single thread.** Per-node search is trivially parallel via rayon; not
  yet wired so the crate stays dep-free.

## Alternatives Considered

* **Filtered-DiskANN edge masking** — simpler, but Wang et al. and our own
  postfilter numbers both show recall collapse for narrow ranges. Rejected.
* **Milvus-style partition keys** — coarse, schema-bound, doesn't compose
  with arbitrary range predicates. Rejected.
* **ACORN (Patel 2024)** — predicate-agnostic graph rebuild. Strong for
  categorical predicates but heavier engineering and not range-specific.
  Tracked as a future backend behind the same `RangeAnn` trait.
* **Compressed SeRF (Zuo & Deng 2024) as v0.1** — superior memory but the
  edge-interval bookkeeping is substantially more code and is best landed
  after the simple-segment baseline is in tree to A/B against. Tracked as
  roadmap item 1.

## References

* `crates/ruvector-serf/` — implementation
* `docs/research/nightly/2026-05-25-serf-range-filter-ann/README.md` — full
  research write-up, SOTA survey, benchmark methodology, roadmap
* Zuo & Deng, VLDB 2024 — SeRF
* Xu et al., SIGMOD 2024 — iRangeGraph
* Patel et al., SIGMOD 2024 — ACORN
* Wang et al., WWW 2023 — Filtered-DiskANN
