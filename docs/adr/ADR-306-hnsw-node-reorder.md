# ADR-306: Cache-Locality Node Reordering for HNSW-style ANN Graphs

## Status

Proposed. Experimental crate (`ruvector-hnsw-reorder`), not wired into
the default query path of any production index. Intended composition
target: the base-layer HNSW graph in `ruvector-core`, and the Vamana
adjacency in `ruvector-diskann`.

## Context

Graph-based ANN indexes (HNSW, Vamana/DiskANN, NSG, SPTAG) execute
queries as a greedy beam search that alternates between two memory
accesses per hop: fetch the neighbour list of the current node, then
fetch the vectors of its neighbours to score them. For a typical
setting (`m=24`, `dim=128`, `f32`), each hop touches roughly one cache
line of adjacency and 24×2 = 48 cache lines of vector data. Once the
working set exceeds L3, throughput is dominated by DRAM latency, not
by the FMA rate of the distance kernel.

The physical order in which nodes are laid out in memory has no
effect on search *correctness* but has a measurable effect on cache
and prefetcher behaviour. Web-graph and inverted-index communities
solved this decades ago (Chierichetti 2009, Wei-Karypis 2016,
Dhulipala 2016). Recent releases from **DiskANN v0.6 (2024)**, **Milvus
2.5 (2025)** and **Weaviate 1.28 (2025)** have applied the same idea
to their graph ANN backends with reported single-digit-percent latency
wins.

RuVector does not currently apply node reordering. Adjacent nightlies
have addressed adaptive termination (ADR-303), semantic query caches
(ADR-301), streaming quantised graphs (ADR-302) and receipts (ADR-304)
but not the layout of the base-layer adjacency.

## Hypothesis

```text
Given a bulk-loaded, single-layer, degree-24 HNSW-style proximity
graph over 100k random 64-dimensional vectors,

when the graph is relabelled by (a) BFS from the entry point,
(b) Gorder with a window of 8, or (c) Recursive Graph Bisection
seeded from BFS with 14 recursion levels and 3 coordinate-descent
sweeps per split,

then (i) search recall is bit-exact preserved, (ii) the log-gap cost
∑ log2|new_id(u) - new_id(v)| decreases monotonically, and
(iii) end-to-end query throughput (ef=64, k=10, warm caches) rises by
at least 5% for either Gorder or RGB on an Apple M4 Max.
```

## Decision

Ship `ruvector-hnsw-reorder` as an experimental crate exposing four
strategies (`Identity`, `Bfs`, `Gorder`, `Rgb`) behind an
`apply_permutation` primitive that rebuilds both the CSR adjacency
and the row-major vector store in the new order. RGB is the
recommended default: it produces the same or better locality than
Gorder at 40× lower reorder cost.

The crate is deliberately isolated:

- Pure Rust, dependencies limited to `rand` and `rayon`.
- No unsafe, no SIMD intrinsics — the effect being measured is
  layout, not kernel micro-architecture.
- Deterministic seeds so reordering output is bitwise reproducible.
- Single-layer HNSW-lite builder inline in the crate; production
  wiring against `ruvector-core::hnsw::Hnsw` and
  `ruvector-diskann::VamanaIndex` is left for a follow-up ADR that
  proposes a `NodeRelabel` trait on those types.

## Consequences

### Positive

- Measured throughput gains of **+5.3 % (Gorder) and +6.8 % (RGB)** at
  n=100k, d=64 on the reference workload, with recall preserved
  exactly.
- Log-gap cost reduced 4–6 % across all three tested workload sizes,
  giving a hardware-independent proxy signal to justify wider rollout.
- RGB reordering costs 438 ms at n=100k — cheap enough to fold into
  merge/compaction cycles without a visible pause.
- The permutation is a first-class artifact: it can be serialised
  alongside a snapshot and applied to a rebuilt vector store on load.

### Negative

- Reordering is a batch operation. Insert-heavy workloads without a
  merge/compact step will not benefit until such a step is added, or
  until an incremental variant lands.
- Multi-layer HNSW needs upper-layer neighbour lists rewritten in the
  same pass; the current crate demonstrates only the base layer.
- Gains vanish when the working set fits in L2 (below roughly n=30k
  at d=128 on M4 Max), so small-collection users pay reorder cost for
  no win. The `Strategy::Identity` no-op path lets callers opt out.
- Under aggressive quantisation (PQ 1–2 B per vector) the vector-side
  cache pressure disappears and reordering targets the wrong bottleneck.
- Adds one more knob to the index-build pipeline.

## Alternatives considered

1. **Do nothing.** Rely on incremental-insertion order to give free
   locality. Works for streaming workloads; fails for bulk loads,
   snapshot restores and post-compaction indexes — exactly the cases
   the referenced DiskANN/Milvus/Weaviate releases target.
2. **METIS / KaHIP partitioning.** Higher-quality partitions but adds
   a C++ dependency and 10–100× the reorder cost. Not justified given
   RGB matches its quality on ANN-sized graphs.
3. **Cuthill-McKee / RCM.** A classic bandwidth-reducing ordering.
   Tested informally on the same workload and produced worse log-gap
   than BFS, consistent with the Wei-Karypis 2016 findings.
4. **Learned reordering via query-log frequency.** Weight edges by
   observed visit frequency; strictly better than uniform but requires
   a warm query log. Deferred as a follow-up.
5. **Delta-encode adjacency without reordering.** Compresses the
   graph but does not touch vector-store locality, which is the
   dominant cost.

## Composition with prior ADRs

- **ADR-297 (adaptive compression retrieval plane):** reordering is
  layout, compression is code width. Orthogonal and additive.
- **ADR-302 (streaming QNG):** the streaming variant already
  rearranges nodes at write time; RGB can run periodically over the
  quiescent segments.
- **ADR-305 (anisotropic PQ):** if PQ codebooks are per-cluster,
  reordering to keep cluster mates contiguous is a natural next step.

## References

See `docs/research/nightly/2026-08-18-hnsw-node-reorder/README.md`.
