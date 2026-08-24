# ADR-340 — Cache-conscious HNSW node ordering

- **Status**: Proposed (nightly research 2026-08-24)
- **Deciders**: RuVector core (nightly cycle)
- **Date**: 2026-08-24
- **Related**: ADR-004 (HNSW), ADR-160-family (delta/graph), nightly `hnsw-repair`, `centroid-seeded-hnsw`

## Context

For every HNSW / NSG-style query, the greedy beam probes the neighbors of the
current candidate, computes distances to their vectors, and pushes surviving
candidates onto a heap. On a corpus of $n$ vectors of dimension $d$ with graph
degree $R$, one query does roughly `ef * R` distance evaluations and touches
`ef * R` neighbor lists.

Both accesses are **random** in the current implementation: nodes are stored
in insertion order, and a neighbor's ID has no relationship to the current
node's ID. Once $n \cdot d$ bytes exceeds L2 (a 100k × 128-dim f32 corpus is
51 MB — already spilling into L3/DRAM on Apple M-series), each neighbor probe
pays a DRAM latency hit that dominates the ALU cost of the dot product.

Sparse-matrix numerics has known this problem for 60 years: reorder the rows
so that non-zeros cluster near the diagonal (bandwidth reduction) and the
matrix-vector product becomes cache-friendly. HNSW's adjacency structure is
exactly such a sparse graph. We haven't been applying the trick.

Recent related work: DiskANN reorders vertices along disk pages for I/O;
CAGRA (GPU) benefits from coalesced accesses via node clustering; Google's
ScaNN uses AH-quantised layouts. None of these are what we ship today, and
nothing in the nightly research history covers **layout-only reordering**
of an existing graph.

## Decision

Add a small `ruvector-cache-conscious-hnsw` crate that:

1. Represents the graph as flat SoA arrays (`vectors: Vec<f32>`,
   `neighbors: Vec<u32>`, `neighbor_counts: Vec<u32>`).
2. Defines a `NodeOrdering` trait returning a permutation
   `old_id -> new_id`.
3. Ships three implementations: `Insertion` (baseline), `Bfs` (from entry
   point), `ReverseCuthillMcKee` (BFS with ascending-degree child sort).
4. Provides `apply_permutation()` and `reorder_with()` helpers that
   materialise a reordered `FlatGraph` in $O(n \cdot (d + R))$.

Reordering is a pure graph isomorphism — distances and recall are preserved
bit-exactly. The **only** thing that changes is the memory-access order of
the beam search.

## Consequences

Positive:

- Measured **1.17× (BFS) — 1.29× (RCM) speedup** at n=100k, dim=128, ef=128,
  same recall (0.286). At n=200k, dim=128 the gain is ~1.10×, still
  recall-neutral.
- Zero recall risk: the trait is a permutation, not a rebuild.
- Composable: works with any existing HNSW backend that exposes an
  edge list.
- Reordering cost is a one-shot $O(n(d + R))$ pass — 27 ms for BFS on 100k
  nodes; amortised over millions of queries.

Neutral / negative:

- Only helps when working-set > L2. Small corpora (< 8 MB) see near-zero
  gains.
- Random-Gaussian data benefits less than clustered embedding data; the
  benchmark uses a mixture-of-Gaussians corpus (64 clusters) as a
  realistic-but-worse-than-real-embedding proxy. Production BERT/OpenAI
  embeddings are far more clustered and should see larger gains.
- Graphs built with mutations (deletes, hnsw-repair) need re-reordering
  after significant churn.

## Alternatives considered

- **METIS / kaHIP k-way partitioning**: theoretically optimal bandwidth
  reduction but adds a C++ dep and multi-second build cost. Deferred to a
  future ADR if 1.29× RCM isn't enough.
- **Space-filling-curve ordering (Hilbert/Morton)**: works when vectors have
  a natural spatial embedding (< 8-dim), not for 128+ dim.
- **Learned orderings** (train a tiny GNN to predict per-node hotness):
  interesting future direction; slotted behind the `NodeOrdering` trait.
- **Do nothing**: the status quo — every query pays random-access DRAM cost.

## Rollout

- Phase 1 (this ADR): standalone crate + bench.
- Phase 2: expose `.reorder(strategy)` on `ruvector-core::HnswIndex` once
  API stability lands.
- Phase 3: opportunistic auto-reorder after N mutations, gated by a
  span-metric threshold.
