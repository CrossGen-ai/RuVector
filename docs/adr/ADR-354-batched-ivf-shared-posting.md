# ADR-354: Multi-Query Batched IVF Scan with Shared Posting Lists

**Status:** Proposed
**Date:** 2026-10-04
**Related:** ADR-353 (BlockMax-IVF scan), ruvector-pq-fastscan-lut16

## Context

Ruvector's IVF query path currently handles concurrent queries by
running one `search()` call per query, even when many of the queries
in a short time window probe the same centroids. Posting-list memory
is re-fetched per query, so throughput at moderate batch sizes is
bandwidth-bound.

Three forces push toward a batched scanner:
1. Agent-memory workloads (ADR-349, ADR-351) submit short bursts of
   related retrievals that hit heavily overlapping centroids.
2. Edge deployments (ADR-351) share one small CPU across many tenants
   where per-query bandwidth is at a premium.
3. Downstream rerankers (GNN rerank, hubness rerank) take batches as
   input anyway, so there is no latency cost to assembling a batch.

## Decision

Introduce a `ruvector-batched-ivf` crate that offers three
semantically equivalent scanners against a shared `IvfIndex`:

- **Serial** — one query at a time (baseline, kept for A/B).
- **List-major batched** — invert the probe plan to
  `list_id -> [query_id]`, scan each list once, update every
  subscribing query's top-k.
- **Tiled batched** — walk the posting list in tiles of 32 vectors so
  the inner `(query, vector)` loop keeps each vector resident in L1.

Correctness is identical across the three (verified by a seeded
`all_strategies_agree` test). Choice of strategy becomes a runtime
hint based on batch size and list density.

## Consequences

Positive:
- Measured 1.2–1.8× latency reduction on 50k × 128-D IVF
  (`n_probe=16`), biggest at the `bs=16` sweet spot. Numbers in the
  companion research doc, reproducible via `cargo run --release
  --example demo`.
- Composes with existing per-list optimizations (BlockMax upper
  bounds, PQ-FastScan LUT16); nothing in the batched path changes
  what a per-list scanner looks like, only how often it is called.
- No new unsafe surface; crate is <400 LoC.

Negative:
- Batched scanners carry one top-k heap per query, so RAM for the
  heap grows linearly with batch size. At `bs=256` memory pressure
  starts to erode gains.
- Latency floor for a single query inside a batch is bounded below by
  the posting-list walk, so very large batches can hurt tail latency
  for the earliest-arriving query. Caller must bound batch size.
- Multi-threaded version requires partitioning subscribers by query
  id to keep heap updates lock-free; this ADR leaves that as follow-up
  work.

## Alternatives considered

1. **Keep serial scanning, add BLAS over centroid distances.** Only
   helps the planning step (which already costs <1 % of total time).
   Rejected: addresses the wrong hotspot.
2. **GPU batched search (BANG-style).** Significant engineering lift;
   deferred until Ruvector's GPU runner has a stable kernel surface.
3. **Query-level pipelining without inversion.** Fetches each list
   once per thread, but relies on OS prefetchers and loses the inner
   tiling win. Benchmark showed <1.1× over serial.

## Rollout

1. Land `crates/ruvector-batched-ivf` (this ADR + branch
   `research/nightly/2026-10-04-batched-ivf-shared-posting`).
2. Wire the batched scanners behind a `BatchedScanner` trait in
   `ruvector-core` so IVF/PQ/BlockMax indexes can advertise support.
3. Add a Rayon-parallel variant and benchmark at `bs ≥ 128` on a
   production corpus.
4. Deprecate per-query scan in the agent-memory server (ADR-349).
