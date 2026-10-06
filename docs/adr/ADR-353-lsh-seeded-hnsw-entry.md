# ADR-353: LSH-seeded entry-point selection for small-world ANN graphs

**Status:** Proposed (promotion-ready)
**Date:** 2026-10-06
**Related:** ADR-346 (deterministic MinCut partitions), ADR-352 (Lloyd-Max SQ4)

## Context

HNSW and NSW indexes greedy-walk from a fixed enterpoint. On hard or
cross-cluster queries the entry is far from the true nearest neighborhood,
so the beam burns 50-100 hops crossing the graph before it localizes.
This is the dominant tax on p99 search latency when the dataset has strong
cluster structure (common for RAG embeddings, product catalogs, geo-tagged
media).

Two prior ruvector improvements touch the same path:
- `ruvector-centroid-seeded-hnsw` seeds from IVF centroid results (needs
  a trained IVF index alongside the graph).
- `ruvector-hnsw-prefetch` reduces per-hop cost but not hop count.

We want the hop-count win without the structural cost of maintaining a
second index.

## Decision

Add a 32-96 bit signed random-hyperplane LSH signature per node, computed
once at build time (O(n·b·d) FMA, 1-2 % of graph-build cost). Bucket all
nodes by their code into a `HashMap<u64, u32>` keyed on the code, value
the bucket representative. At query time the signature is computed on the
query (O(b·d) FMA, ~1 µs for b=12, d=96) and used as the entry seed;
empty buckets fall back to the fixed enterpoint, so recall is bounded
below by the baseline.

Shipping as a standalone crate `ruvector-lsh-seeded-hnsw-entry` with a
pluggable `EntryStrategy` enum so existing HNSW crates can opt in without
API churn.

## Consequences

**Positive**
- On a pure exact-kNN proximity graph (no long edges) LSH-seeded entry
  moves recall@10 from 3.1 % to 95.1 % at ef=64. On a small-world graph
  (4 random long edges per node, m=24), recall is unchanged (97.1 %) and
  per-query latency drops 45 % (114 → 63 µs), distance comps drop 19 %
  (760 → 614).
- Memory overhead: 8 B/node (one u64 per signature) + a small
  `HashMap<u64, u32>` sized at ~`min(2^b, n)`. For n=1 M, b=16 that is
  ~16 MB of signatures + at most 2 MB of map — negligible next to a
  typical HNSW link table.
- No training needed: hyperplanes drawn i.i.d. standard normal, seeded.
  No periodic retraining, no coupling to data distribution.
- Build cost: 1-2 % of HNSW build time. Query cost: 12-96 FMA before
  the greedy walk, swallowed by the first hop's distance computation.

**Negative**
- LSH buckets are not perfect. Measured bucket population on a 32-cluster
  synthetic dataset: 12-bit signature produced 518 non-empty buckets out
  of 4096. Some buckets map to a representative in the wrong cluster;
  these queries fall back to the fixed enterpoint's behavior.
- Signature quality degrades on very high-dim data where sign-only
  codes under-sample angular structure. ADR-347 OPQ rotation and the
  RaBitQ family dominate here — this crate is complementary, not a
  replacement.

## Alternatives considered

1. **Centroid-seeded entry** (prior art): needs an IVF index and incurs
   centroid-scan cost at every query (~`k_coarse` distance comps vs
   `b` FMA). Rejected as default; still available via a separate crate.
2. **Multi-entry sampling**: start the walk from k random nodes and
   union the beams. Costs k·ef_entry extra distance comps for no
   deterministic win. Already evaluated in `ruvector-hnsw-rng-diverse`.
3. **Learned entry via tiny MLP**: 1-2 pt recall over LSH in a
   literature survey (ScaNN "GumbelAnn"), but needs training data and
   introduces a non-stationary dependency on the data distribution.
   Deferred.

## Promotion plan

1. Nightly research branch (this ADR + the `ruvector-lsh-seeded-hnsw-entry`
   crate + benches).
2. Fold the `EntryStrategy` trait into `ruvector-coherence-hnsw` behind
   a `lsh-seed` cargo feature.
3. Public bench against `ruvector-sota-bench` on SIFT1M + Deep1B
   subsamples.
4. Promote to `on-by-default` in the edge path once latency parity on
   small-world graphs is reproduced on the ruvector CI (M3, Linux x86).

## References

See `docs/research/nightly/2026-10-06-lsh-seeded-hnsw-entry/README.md`.
