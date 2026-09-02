<!-- Also numbered ADR-343 per the docs/adr/ADR-3XX repo convention;
     scaffolder filed as 0001 because its registry did not see the
     ADR-3XX series. Cross-reference from the nightly research README. -->

# ADR-343: Cache-Oblivious HNSW Graph Layouts (BFS / DFS / vEB)

> Decision date: 2026-09-02
> Status: Proposed (research crate only, not wired into `ruvector-hnsw`)
> Scope: `crates/ruvector-cache-oblivious-hnsw` — standalone workspace
> Drivers: HNSW greedy-search latency is memory-bound at 128-d f32;
>          static node layout is the untouched lever.

## Context

HNSW greedy search on a 128-d f32 corpus is a pointer-chasing workload.
Distance kernels are cheap on M-series and modern x86; wall-clock
latency is dominated by cache and DRAM traffic to fetch the next vector
and adjacency list. Every in-tree HNSW implementation we surveyed
(`ruvector-hnsw`, hnswlib, FAISS-HNSW, Milvus 2.4, Qdrant 1.11) stores
nodes in insertion order — the natural "BFS-shaped" order that a builder
produces. Milvus 2.4 added a graph-reorder pass, but only BFS.

The cache-oblivious algorithms literature (Frigo et al. 1999; Bender et
al. 2002) gives an `O(log_B n)` amortized memory-transfer bound on
root-to-leaf tree walks under a van Emde Boas (vEB) recursive layout,
independent of cache-line size `B`. HNSW's greedy walk is not exactly a
tree walk, but the frontier is concentrated near the target and shares
ancestor visits across queries, so the guarantee should transfer to
first order. We could not find any HNSW implementation, paper, or
benchmark that measures the effect systematically.

## Decision

Ship a standalone research crate,
`crates/ruvector-cache-oblivious-hnsw` (its own `[workspace]` so the
ruvector workspace build is unaffected), with:

1. `FlatGraph { layout, vectors, neighbours, m, perm, inv, entry }` —
   a contiguous permutable HNSW graph.
2. `Layout::{Bfs, Dfs, Veb}` plus one permutation function per variant
   (`bfs_permutation`, `dfs_permutation`, `veb_permutation`).
3. A small greedy HNSW-style builder (`build_hnsw`) that produces a
   logical graph, then materializes it under a chosen `Layout`.
4. A layout-agnostic `greedy_search`, counting visited nodes and
   distance evaluations so per-layout comparison is exact.
5. Unit tests: permutation bijectivity per layout, top-1 agreement of
   the three layouts on stored queries, non-empty results.
6. A runnable end-to-end benchmark (`examples/bench.rs`) reporting
   real per-query latency, QPS, and a slot-stride locality proxy.

## Alternatives Considered

- **DiskANN block layout** — coarser, disk-oriented; complementary but
  does not deliver the cache-line-size-independent guarantee.
- **Insertion-order + LRU hint** — cheap to maintain, no help for queries
  targeting non-recent nodes.
- **Learned permutation from query traces** — high complexity, no static
  guarantee. Deferred.
- **Block-vEB** — align micro-block boundaries to 64-B cache lines.
  Listed as next-step in the research README, not shipped here.

## Consequences

Positive:
- vEB measured at **−18.8 % latency, +23.1 % QPS** versus BFS on
  M4 Max, N=50k, dim=128, k=10, ef=64.
- Visited/dist counts are byte-identical across layouts, cleanly
  isolating cache traffic as the causal variable.
- Standalone crate keeps the workspace build untouched.

Negative:
- Layout is static; any insert or delete invalidates it. Production
  HNSW needs incremental relayout or periodic compaction, neither of
  which is in this crate.
- The bundled HNSW builder is a small single-layer greedy graph, not
  `ruvector-hnsw`. Numbers will differ on the production index; the
  layout effect should still transfer.
- Single-threaded measurement only. Multi-thread contention on a shared
  L2 may attenuate the win.
- The "average slot stride" locality metric moved in the *opposite*
  direction from latency — a reminder to validate with real perf
  counters before productionizing.

## Testable Criteria

| ID   | Criterion                                                              | How verified                                                       |
|------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| TC-1 | All three layouts build without error at N=800.                         | `cargo test --release -p ruvector-cache-oblivious-hnsw` (pass)     |
| TC-2 | Permutation is bijective for each layout.                               | `permutation_is_a_bijection_all_layouts` unit test (pass)          |
| TC-3 | Top-1 agrees across BFS/DFS/vEB on ≥25/30 stored queries.               | `layouts_return_identical_top1_for_stored_queries` (pass)          |
| TC-4 | vEB layout reduces mean per-query latency by ≥10 % vs BFS baseline.     | `cargo run --release --example bench` → 18.8 % (see research README) |
| TC-5 | Visited-node count is identical across layouts (isolates cache effect). | Benchmark output: 1153.6 visited/q for all three                   |

## References

- Frigo, Leiserson, Prokop, Ramachandran. *Cache-Oblivious Algorithms.*
  FOCS 1999.
- Bender et al. *Two Simplified Algorithms for Maintaining Order in a
  List.* ESA 2002.
- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs.* IEEE TPAMI
  2020 (arXiv:1603.09320).
- Subramanya et al. *DiskANN.* NeurIPS 2019.
- Gao et al. *RaBitQ.* SIGMOD 2024 (arXiv:2405.12497).
- Nightly research README:
  `docs/research/nightly/2026-09-02-cache-oblivious-hnsw-layout/README.md`
