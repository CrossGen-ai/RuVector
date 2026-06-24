# ADR-268: Hubness-Aware HNSW (anti-hub pruning)

**Status**: Proposed
**Date**: 2026-06-24
**Authors**: ruvector nightly research agent
**Related**: ADR-264 (Matryoshka coarse-to-fine), ADR-265 (Benchmark suite), ADR-267 (SOTA validation)

---

## Context

RuVector's graph-based ANN indices (HNSW family, NSG-style variants) inherit
the **hubness phenomenon** of high-dimensional vector spaces: a small fraction
of nodes accumulate disproportionate *incoming* edges. The HNSW paper's
`selectNeighborsHeuristic` bounds *outgoing* degree at `M_max`, but the
bidirectional insertion path leaves indegree unbounded.

On a 5,000 × 64-d synthetic Gaussian benchmark with `M=16`, the baseline NSW
produces:

- mean indegree: 22
- **max indegree: 982** (≈ 45× the mean)
- 6.5% of nodes are "hubs" (indeg > 3 × mean)
- Gini coefficient: 0.58 (heavy right tail)

These hubs:
1. Inflate p95 latency (long neighbour lists at search time).
2. Waste edge memory (one tail node holds ~1% of total edges).
3. Distort greedy traversal toward the hubs rather than the true geodesic.

No open-source vector DB (FAISS, hnswlib, Qdrant, Weaviate, Milvus,
Pinecone-OSS) exposes a deterministic post-build indegree cap with published
recall/latency/memory trade-off measurements.

## Decision

Adopt **anti-hub pruning** as a first-class, optional post-build pass on
graph-based RuVector indices, exposed via a `IndegreeCap` policy:

| Policy     | Cap       | Use case                                           |
|------------|-----------|----------------------------------------------------|
| `None`     | unbounded | Current default; preserved for backward compat.    |
| `Light`    | `3 · M`   | Default *recommendation* for prod; trim long tail. |
| `Aggressive` | `2 · M` | Memory-constrained or tail-latency-critical paths. |

The pass:

1. Builds reverse adjacency (O(E)).
2. For each node with `indeg > cap`, sorts its incoming sources by distance
   ascending and keeps the closest `cap`.
3. Drops the rest, subject to an under-degree guard: an edge is not removed
   if its *source* would fall below `max(M/2, 2)` outgoing edges.

A reference implementation lives in `crates/ruvector-hub-hnsw/` and is wired
into the workspace as a feature-gated module. Production HNSW (`ruvector-core::hnsw`)
will gain a method `anti_hub_prune(cap: IndegreeCap)` in a follow-up iter.

## Consequences

**Positive (measured on 5K × 64-d, real `cargo run --release`):**

- Max indegree drops 982 → 65 (**15× reduction**) under both cap policies.
- Recall@10 loss bounded at **−0.4 pp** (0.9155 → 0.9115 at Aggressive).
- Edge count drops 25–32% → proportional RAM savings for adjacency arrays.
- QPS +4.6%, p95 latency −4.6% at Aggressive cap.
- Indegree-distribution Gini drops 0.58 → 0.42; hub fraction 6.5% → 1.0%.

**Negative / risks:**

- Aggressive cap on legitimately-hub-routed clusters (mega-categories in
  e-commerce embeddings) may disconnect a few sources from a remote cluster.
  Mitigation: the under-degree guard, plus a post-prune
  weakly-connected-component check before promoting an index.
- Streaming-write workloads will re-grow hubs between passes. Mitigation:
  trigger pass when `max_indeg / mean_indeg > τ` (proposed default τ = 5).
- Adds an O(E + N · cap · log(cap)) build-time cost (≈ 1 ms on the PoC
  benchmark — negligible vs. the 286 ms NSW build itself).

**Operational:**

- Default policy stays `None` to preserve backward compatibility.
- Documented as a `--hub-cap {none|light|aggressive}` flag on the CLI and
  index-config struct.
- Indegree-distribution stats (`max`, `p99`, `gini`, `hub_fraction`) become
  part of the standard benchmark output (`ruvector-bench`).

## Alternatives Considered

1. **Reverse k-NN sub-sampling at construction** (Hara et al. 2023). Strictly
   more powerful but requires intrusive changes to the insert path, breaks
   the existing HNSW build API, and roughly doubles build time. Rejected for
   the first iteration; revisit if the post-build pass proves insufficient
   on real datasets.
2. **Edge re-routing instead of edge dropping.** Replaces a hub edge with an
   edge to the next-nearest non-hub node. Preserves connectivity but costs
   one extra distance computation per pruned edge and complicates the
   under-degree guard. Filed as roadmap item #3.
3. **Increase `M` globally instead.** Larger `M` smears the indegree
   distribution but does not change its skew — the max-indegree-to-mean
   ratio is roughly scale-invariant in `M`. Rejected on theoretical grounds.
4. **Random edge drop on hubs.** Empirically equivalent to the "drop
   furthest" heuristic on uniform Gaussian data but worse on
   cluster-structured data because it can sever the legitimate close-source
   edges that keep the cluster reachable. Rejected.

## Rollout Plan

1. **Now (this ADR):** Ship `crates/ruvector-hub-hnsw/` as a measurement and
   reference crate. No change to default index behaviour.
2. **+1 iter:** Add `anti_hub_prune` method to `ruvector-core::hnsw::Index`,
   gated behind `config.hub_cap = Light`. Keep default `None`.
3. **+2 iter:** Run the same protocol on SIFT-1M / GIST-1M / MS-MARCO under
   the SOTA validation protocol (ADR-267) and report a signed manifest.
4. **+3 iter:** If real-dataset results match the PoC, flip default to
   `Light` in a major-version release with a clearly-documented opt-out.
