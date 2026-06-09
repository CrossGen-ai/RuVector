---
adr: 199
title: "Tribase IVF — Triangle-Inequality Window Pruning for Inverted-File ANN"
status: accepted
date: 2026-06-09
authors: [ruvnet, claude-nightly]
related: [ADR-193]
tags: [ivf, ann, vector-search, tribase, triangle-inequality, pruning, nightly-research]
---

# ADR-199 — Tribase IVF: Triangle-Inequality Window Pruning

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-06-09-tribase-ivf-pruning` as
`crates/ruvector-tribase`. `cargo build --release -p ruvector-tribase`
succeeds; `cargo test -p ruvector-tribase` runs 9 tests, all green;
`cargo run --release -p ruvector-tribase --bin tribase-demo` produces
the benchmark numbers cited below.

## Context

`ruvector` already has multiple IVF flavours (`ruvector-rairs`,
`ruvector-betivf`, `ruvector-soar`, plus the PQ family in
`ruvector-opq` / `ruvector-anisotropic-pq`). All of them, at query
time, share the same fundamental inner loop: *scan every member of
every probed posting list and compute a full distance*. That inner
loop is what limits IVF latency in production — the rest of the
pipeline (centroid scan, heap maintenance) is small.

Prior pruning techniques inside the inner loop are mostly
*approximate*: PQ, RaBitQ, LeanVec all trade recall for speed. For
workloads that require exact answers — billing audits, e-discovery,
provenance verification — the only knob available is reducing
`n_probe`, which directly costs recall.

Tribase (Liu et al., SIGMOD 2024) showed that the triangle
inequality, applied with a single pre-sorted distance per point,
can prune **>85% of inner-loop candidates exactly**, with no recall
penalty and ~1.5% memory overhead. This ADR adds Tribase as a
first-class IVF backend in ruvector.

## Decision

We ship `crates/ruvector-tribase` containing three indices behind a
common `AnnIndex` trait:

* `FlatIndex` — exhaustive baseline.
* `PlainIvfIndex` — IVF with no pruning (control).
* `TribaseIndex` — IVF with per-cluster posting lists sorted by
  `d(x, c)` and an O(log n) triangle-inequality window plus
  lower-bound short-circuit at query time.

The pruning is *algebraic* — `|d(q,c) − d(x,c)| ≤ d(q,x)` is exact
for any metric — so the index returns identical results to
`PlainIvfIndex` for the same probe set. Recall is preserved by
construction.

### Why this design over the alternatives

| Alternative | Why rejected |
|---|---|
| **PQ-style code DC** | Lossy. Already in `ruvector-opq`; we want an exact path. |
| **SOAR spilling** | Doubles index size. Helps recall, not latency. |
| **Bandit early-stop** | Stochastic; defeats SIMD. |
| **Per-cluster KD-tree** | Higher build cost, worse cache behavior, marginal gains. |

Tribase is the only known IVF inner-loop accelerator that is
simultaneously exact, almost-free in memory, and friendly to
batched scans.

## Consequences

### Positive

* **5.42× query speedup** at `n_probe=16` over `PlainIvfIndex` on
  the shipped benchmark, with recall@10 = 1.000.
* **Speedup grows with `n_probe`**, inverting the usual IVF
  recall-latency tradeoff: extra probes are nearly free because the
  heap has already tightened.
* **Storage cost: +4 bytes per point** (~1.5% on the benchmark).
* Trait-based design (`AnnIndex`) keeps the seam clean for future
  Tribase+RaBitQ, Tribase+SIMD, and Tribase-Disk variants.
* `SearchStats` exposes pruning counts so the speedup story is
  auditable, not narrated.

### Negative / Risks

* Adversarial / out-of-distribution queries (very large `qd`)
  collapse the window and reduce Tribase to plain IVF cost. A
  detector + fallback path is roadmap, not shipped.
* Insertion requires re-sorting the affected posting list; the
  current impl is build-once. Streaming updates need either a
  B-tree-style structure or periodic rebuilds.
* At very high dimension (`d > 512`) the
  centroid-distance distribution concentrates, narrowing both the
  pruning benefit and the SIMD opportunity. Best fit is the
  64–256d range of modern embedding models.

### Neutral

* k-means quality matters: poor centroids produce wide
  centroid-distance distributions and weaker pruning. Same as for
  every other IVF variant.

## Alternatives Considered

1. **Skip the sorted-window step, keep only the lower-bound
   filter.** Simpler, but loses the binary-search head-start and
   degrades to ~30% pruning instead of 85%+.
2. **Use squared distances throughout, including the window
   compare.** Tempting, but `|qd² − xd²|` is not a valid lower
   bound on `d(q,x)²` — the triangle inequality is on distances,
   not their squares.
3. **Combine with PQ asymmetric DC as the primary distance
   estimator.** Deferred to a follow-up crate
   (`ruvector-tribase-rabitq`). Out of scope for this nightly.

## Implementation

* `crates/ruvector-tribase/src/lib.rs` — trait, helpers, k-means.
* `crates/ruvector-tribase/src/ivf.rs` — `FlatIndex`,
  `PlainIvfIndex`.
* `crates/ruvector-tribase/src/tribase.rs` — `TribaseIndex` plus
  unit tests that assert recall equivalence with `PlainIvfIndex`
  and strict pruning dominance.
* `crates/ruvector-tribase/src/main.rs` — `tribase-demo`
  benchmark binary used to produce ADR + research-doc numbers.

## Benchmark Snapshot (50 000 × 64d, k=10, single thread)

| n_probe | plain μs/q | tribase μs/q | speedup | tribase pruned % | recall@10 |
|--------:|-----------:|-------------:|--------:|------------------:|----------:|
|       4 |       32.3 |         18.4 |  1.76×  |             50.8% |    1.000  |
|       8 |       55.9 |         18.7 |  2.99×  |             73.9% |    1.000  |
|      16 |      104.6 |         19.3 |  5.42×  |             86.7% |    1.000  |

Reproduce with
`cargo run --release -p ruvector-tribase --bin tribase-demo`.

## Provenance

The "Tribase" name and SIGMOD 2024 attribution are taken from the
paper *Tribase: A Triangle-Based ANN Search Framework over Vector
Embeddings* (Liu, Xu, Lian, Chen). The implementation here is an
independent re-derivation from the triangle inequality and was not
copied from any reference code. Evaluate against the numbers in
this ADR and the research README, not against the citation.
