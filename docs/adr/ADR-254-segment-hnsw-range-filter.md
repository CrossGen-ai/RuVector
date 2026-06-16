---
adr: 254
title: "Segment-HNSW for Range-Filtered ANN Search"
status: proposed
date: 2026-06-16
authors: [crossgen-ai, claude-nightly]
related: [ADR-253]
tags: [ruvector, ann, range-filter, hnsw, irangegraph, segment-tree, recall, retrieval]
---

# ADR-254 — Segment-HNSW for Range-Filtered ANN Search

## Status

**Proposed.** Working PoC ships in this branch as `crates/ruvector-segment-rangehnsw`. Promotion to a default-on retrieval path is gated on (a) shared vector storage to drop the 7.9× memory overhead, (b) parallel build, and (c) SIFT-1M validation. This ADR records the architectural decision, evidence, and graduation criteria.

## Context

ruvector already ships ACORN (`crates/ruvector-acorn`) for predicate-agnostic filtered HNSW search. ACORN handles **categorical** predicates well: at 1% selectivity it kept 97% recall while running orders of magnitude faster than brute force. But the **continuous range filter** workload — "vectors with timestamp ∈ [T₀, T₁]", "products with price ∈ [P_lo, P_hi]" — has weaker behavior:

- A plain HNSW + post-filter is recall-unstable. We measured 6.5% recall@10 on 1%-selective ranges (n=10K, D=64, k=10, ef=80).
- ACORN-style pre-filter expansion keeps recall (99.5%) but balloons to ~140× brute-force latency, because the graph wasn't built with the filter in mind.
- Brute force scales linearly with corpus size, untenable above ~100K points.

Recent SIGMOD 2024 work (iRangeGraph, SeRF) closes this gap by indexing per-range-window graphs at build time. We need an equivalent in ruvector's stack so retrieval pipelines can compose vector similarity with timestamp / price / score / quality / category-as-percentile filters without giving up recall.

## Decision

Adopt a **segment-tree-of-graphs** index, shipped as the standalone crate `ruvector-segment-rangehnsw`:

1. Points are sorted by a single continuous range key at build time.
2. A binary segment tree partitions the sorted array. Every tree node owns a small proximity graph over its slice. Leaves cover `leaf_size = 256` points.
3. Range queries `[lo, hi]` descend the tree, find the O(log n) nodes fully contained in the range, run a beam search inside each, and linearly scan the boundary leaves. Results are merged by global id and truncated to top-k.
4. The crate exposes `SegmentRangeIndex::build` and `::search(query, k, lo, hi, ef)`. Tests assert recall ≥ 0.80 against brute-force ground truth on Gaussian data.
5. The PoC is intentionally *self-contained*: no dependency on `ruvector-core` or `ruvector-acorn`. This keeps the new path testable in isolation. The production refactor (graduation criterion) will move the underlying graph implementation to share `ruvector-core`'s graph primitives.

## Consequences

**Positive**

- Recall@10 at 1% selectivity: **1.000** vs 0.065 (post-filter) and 0.995 (pre-filter). Closes the recall hole.
- QPS at 1% selectivity: **230 858** vs 22 502 (post-filter) and 841 (pre-filter). 10× over the next-best variant, 2.2× over brute force.
- Recall stays ≥0.96 across the 1–50% selectivity sweep.
- Architecturally simple — one new crate, well-defined contract, easy to swap.

**Negative**

- **Memory 7.9× vs flat graph** (29.92 MiB vs 3.79 MiB on n=10K, D=64). O(log n) factor; can be reduced to ~2× by shared vector storage.
- **Build time 1.7s vs 0.4s** for the flat graph on 10K points. Acceptable; halves with parallel build per level.
- Adds a new index type to maintain. Mitigation: design is small (≤500 lines), tests passing.
- **High-selectivity regime is dominated**: at 50% selectivity, brute force is faster. A query planner must pick the right index — out of scope for this ADR but tracked as a graduation criterion.

## Alternatives considered

1. **Extend ACORN to handle ranges.** ACORN's predicate-agnostic expansion holds recall but pays the global-graph-walk cost on every query. Numbers in the research doc show 140× slowdown at 1% selectivity. Rejected because it doesn't fix the latency.
2. **IVF-style range partitioning.** Pick fixed range buckets at build time, run HNSW per bucket. Simple, but workload-fragile: queries near bucket boundaries oversample, and shifting query distributions need rebuilds.
3. **SeRF (half-bounded ranges only).** Smaller index, but doesn't support the full closed-range queries we need. Could be a future complement, not a replacement.
4. **Filtered DiskANN / Stitched Vamana.** Designed for categorical predicates, not continuous ranges. Worse fit than iRangeGraph.
5. **Wait and let the workload use post-filter.** Rejected: 6.5% recall is unusable.

## Graduation criteria

The PoC is a research artifact, not a production path. Before this index becomes a default retrieval option:

- [ ] Refactor to share `ruvector-core` storage so memory drops to ≤1.5× flat-graph.
- [ ] `rayon`-parallel build across same-depth nodes.
- [ ] SIFT-1M and Deep-1B numbers in `docs/research/nightly/...`.
- [ ] Selectivity-aware planner that auto-picks segment-HNSW vs flat-graph + post-filter vs brute.
- [ ] Multi-dimensional range support (kd-tree of graphs) for AND-joined attribute filters.
- [ ] Integrate with `ruvector-filter` so categorical and range predicates compose under one API.

Until those land this stays in the workspace as `ruvector-segment-rangehnsw` for benchmarking only.

## References

- iRangeGraph (SIGMOD 2024) — arXiv:2403.13865
- SeRF (SIGMOD 2024)
- ACORN nightly research: `docs/research/nightly/2026-04-26-acorn-filtered-hnsw/README.md`
- Nightly research doc for this ADR: `docs/research/nightly/2026-06-16-segment-hnsw-range-filter/README.md`
- PoC crate: `crates/ruvector-segment-rangehnsw/`
