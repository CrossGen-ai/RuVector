# ADR-196 — Dynamic Exploration Graph (DEG) for Streaming Vector Memory

**Status:** Proposed (nightly research PoC)
**Date:** 2026-05-31
**Crate:** `crates/ruvector-deg`
**Related:** ADR-193 (RaIRS-IVF), prior nightly research on HNSW/NSG/SOAR/SymphonyQG.

## Context

ruvector ships several ANN backends (HNSW via DiskANN port, RaBitQ-quantised
flat, IVF variants, SymphonyQG). All of them target *batch-built* indexes:
insertion costs are amortised by a build phase, and deletion is implemented
either by tombstoning (HNSW) or by full rebuild (IVF). The "agent memory"
workloads we target — chat-history retrieval, tool-output recall, long-running
plan state — produce a stream of insertions interleaved with deletions and
queries. Tombstones bloat memory; full rebuilds break tail latency.

The 2023 paper by Hezel et al. ("Fast Approximate Nearest Neighbor Search with
a Dynamic Exploration Graph using Continuous Refinement", arXiv:2307.10479)
proposes the **Dynamic Exploration Graph (DEG)**, a bounded-degree proximity
graph designed from the ground up for streaming insert + delete with
continuous edge refinement. Reported recall is on par with HNSW on SIFT1M
while supporting in-place deletion without tombstones.

No ruvector crate currently implements DEG. This ADR proposes a PoC.

## Decision

Add `crates/ruvector-deg` implementing a minimal DEG:

* **Bounded out-degree D** (default 24). Every live vertex has exactly `D`
  outgoing edges once the graph exceeds the warm-up size; below that it is
  fully connected.
* **Beam search** with a tunable `eps` parameter (build- and query-time
  beam width).
* **In-place deletion:** when a vertex is removed, every vertex that
  pointed at it is re-searched and its dangling edge is patched in-place
  using the best non-neighbour candidate the search finds. The vacated id
  is pushed onto a free list and reused by the next insert, so vector/edge
  storage does not grow with churn.
* **Edge refinement:** `refine` triangle-improvement passes per insert. Each
  pass picks a random outgoing edge, inspects the neighbour's neighbours,
  and replaces the host's heaviest edge if a strictly shorter candidate is
  found.
* **Trait-based metric:** `Metric::L2Sq` and `Metric::Cosine` are provided;
  the metric is passed by value, so future backends (RaBitQ, LVQ, OPQ) can
  plug in without changing the graph code.

The PoC is deliberately small (~500 lines of `graph.rs` + 70 lines of
`distance.rs`) so it can be folded into other crates later (e.g. as the
in-memory tier of `ruvector-rulake`).

## Consequences

**Positive**

* Real streaming support without tombstones — capacity is constant under
  arbitrary insert/delete churn (verified by `streaming_insert_after_delete_reuses_slots`).
* Recall/latency curve is competitive with HNSW: PoC measured recall@10 =
  0.974 at 118 µs/query (5 k × 64-d unit-sphere uniform).
* Pluggable metric makes it a natural substrate for the quantised-distance
  work tracked in ADR-193 and the nightly RaBitQ/LeanVec branches.

**Negative**

* Deletion is expensive in absolute terms (~3.7 ms/delete on the PoC
  dataset). The cost is `O(refs(dead) · search_cost)`; for hot-path
  agent-memory workloads we will need batched deletion or background
  patching.
* No hierarchy: navigation depends on graph diversity alone. Synthetic
  cluster topologies (32 tight Gaussian blobs) collapse recall to <5%
  unless the inserts produce diverse long-range edges. HNSW's layer
  structure papers over this; we accept the limitation for v0 and propose
  a "DEG + IVF entry-point oracle" follow-up in the research doc.
* No persistence yet — all state is in-memory.

## Alternatives Considered

* **HNSW with tombstones.** Simple, but tombstone bloat is real; the
  Microsoft FreshDiskANN paper (already on a sibling research branch)
  shows a more elaborate compaction protocol is required.
* **Full IVF rebuild on churn.** Sacrifices tail latency; unacceptable for
  agent-memory loops where every turn writes new vectors.
* **GLASS / NavHNSW.** Both are batch-builders; neither targets streaming.
* **Vamana / DiskANN.** Targets disk-resident static indexes; orthogonal.

## Acceptance

* `cargo build --release -p ruvector-deg` succeeds.
* `cargo test -p ruvector-deg` passes 4 integration tests, including a
  recall@10 ≥ 0.80 floor on a 1.5 k random dataset.
* Demo binary `deg-demo` prints real recall/latency numbers for three
  `eps` variants — see research doc for the captured numbers.
