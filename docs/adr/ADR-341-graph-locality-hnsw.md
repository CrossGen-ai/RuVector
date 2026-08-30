# ADR-341: Graph-Locality Storage Reordering for HNSW

**Status:** Proposed (research, nightly 2026-08-30)

**Date:** 2026-08-30
**Owners:** ruvector maintainers
**Tracking:** [`docs/research/nightly/2026-08-30-graph-locality-hnsw/`](../research/nightly/2026-08-30-graph-locality-hnsw/README.md)

## Context

Every mainstream HNSW implementation the crate has audited — hnswlib,
FAISS-HNSW, DiskANN, Milvus, Qdrant, Weaviate, and our own
`crates/ruvector-hnsw` — stores vectors in **insertion order** in a flat
buffer, keyed by node id. Neighbor lists hold neighbor ids, and search
dereferences those ids straight into the buffer. At high recall the
reported bottleneck in HNSW is not the distance kernel — it is the
pointer-chase from each visited node to its neighbors.

An A/B run on our reference workload confirms this. On a 50 000 × 128
synthetic clustered dataset, three physically-permuted copies of the *same*
HNSW graph — identical adjacency, identical entry point, identical query
algorithm — differ in QPS by up to +19 % with **bit-identical recall and
bit-identical distance-call counts**. The only thing changed is the
memory address in which each vector lives.

The two things we already know that reorder-nothing implementations
inherit as costs:

1. Insertion order is uncorrelated with graph proximity, so an edge's mean
   |id gap| grows roughly with dataset size. For 50k×128 it is ~12 226,
   meaning each neighbor dereference reads a vector on average ~6 MiB away
   in the buffer — a guaranteed miss past L2.
2. Rebuilding the index (offline compaction, snapshot restore, cluster
   rebalance) is the natural time to change layout. That opportunity is
   silently discarded today.

## Decision

Introduce a **layout-reordering pass** as a first-class operation on the
HNSW index, expressed through a `ReorderStrategy` trait. The nightly
research crate `ruvector-graph-locality-hnsw` establishes the API and the
measurement methodology; a promoted `ruvector-hnsw-layout` crate will host
production strategies.

The pass is:

- **Pure.** A function of the source index; the input is not mutated.
- **Correctness-preserving.** The permutation is a bijection, and the
  reordered index returns the same set of nearest neighbors for every
  query (checked in-tree by `reordering_preserves_neighbor_set`).
- **Pluggable.** Three baseline strategies land now: Identity, BFS from
  the entry point, and Reverse Cuthill-McKee over the symmetrized L0
  adjacency. Louvain / community and learned layouts follow.
- **Amortized.** Reorder runs at build, snapshot compaction, or scheduled
  maintenance windows — never on the query path.

For the shipped API the reordered index carries a `new_to_old: Vec<u32>`
inverse so callers holding external id references (payloads, tombstones)
can continue to use their existing ids.

## Consequences

**Positive**

- Free QPS. On our reference workload, BFS delivers +17-19 % QPS at
  n = 50 k with identical recall and identical work counts; RCM delivers
  +13-16 %. This is a Pareto win, not a tradeoff.
- Locality metric (`mean_edge_gap`) is hardware-independent and cheap;
  it stays useful as a regression signal in CI when hardware counters
  are unavailable.
- The trait leaves headroom for community-detection and learned layouts
  without touching the query path.
- Reorder is the natural time to compact deleted-node tombstones, which
  currently accumulate in `crates/ruvector-hnsw`.

**Negative**

- Inserts after a reorder land at the tail of the new buffer, physically
  far from their graph neighbors. Any long-running write workload will
  drift back toward the identity-layout regime unless a compaction is
  scheduled. The crate should expose a "layout drift" gauge and a
  documented schedule.
- Snapshot format needs a `layout_hash` so mismatched caches can be
  detected. This is a small format bump but a real one.
- Choosing the strategy is a hyperparameter. BFS wins for entry-point-
  dominated workloads; RCM wins when queries land uniformly over the
  graph. There is no free automatic answer yet; the CLI must expose the
  choice.

**Neutral**

- No change to on-wire distance semantics, quantization, or graph
  construction. Existing ADRs on ACORN, RaBitQ, coherence-HNSW, etc.,
  compose with layout reordering along the vector-storage axis
  orthogonally.

## Alternatives considered

1. **Cache-line-sized neighbor packing (a la DiskANN)** — pack each node's
   neighbor list adjacent to its vector. This reduces one pointer chase
   but does not help the *neighbor's vector* fetch, which is the dominant
   miss. Not mutually exclusive; complementary.
2. **Learned quantization only (RaBitQ, PQ)** — cuts distance-computation
   cost, not neighbor-dereference cost. At high recall the two costs are
   independent; both matter.
3. **Recompute the graph with a locality-aware selector** — heavy, risks
   changing recall. Reordering leaves the graph untouched and is a pure
   engineering pass.
4. **Do nothing.** Leaves a measured 13-19 % QPS improvement on the floor
   with no correctness cost.

## Follow-ups

- Promote the strategies into `crates/ruvector-hnsw` behind a
  `StorageLayout` trait; move heavy strategies (Louvain, learned) into a
  new `ruvector-hnsw-layout` crate.
- Add `ruvector index reorder --strategy {bfs,rcm,...}` to the CLI.
- Wire per-strategy hardware-counter metrics (LLC-miss rate) into
  `SearchStats` so future strategies can be tuned against the real
  bottleneck, not the id-gap proxy.
- Investigate community-detection layouts (Louvain, METIS) for
  hub-dominated workloads.
