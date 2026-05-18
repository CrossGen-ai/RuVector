---
adr: 194
title: "RoarGraph — Projected Bipartite Graph for OOD Cross-Modal ANNS"
status: accepted
date: 2026-05-18
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-155, ADR-193]
tags: [graph-index, ann, vector-search, ood, cross-modal, roargraph, nightly-research]
---

# ADR-194 — RoarGraph: Query-Aware Graph Index for Out-of-Distribution ANNS

> **Provenance note.** The RoarGraph algorithm is attributed to
> Chen et al., VLDB 2024, and a reference arXiv id of 2408.08933 appears in
> the VLDB proceedings listing.  We were **unable to independently verify**
> that this arXiv id resolves to the named paper at the time of writing
> (network fetch was not attempted; the id may be correct but unverified).
> The algorithm described here — projecting a bipartite query–base graph onto
> the base set — is implemented from the paper's published description and
> evaluated on fully reproducible synthetic benchmarks.  Treat it as an
> original implementation of the bipartite-projection idea and judge it on
> `crates/ruvector-roargraph/src/main.rs`, not on the citation.

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-18-roargraph` as
`crates/ruvector-roargraph`.  All 11 unit tests pass; release build is green
with `cargo build --release -p ruvector-roargraph`.

## Context

ruvector has robust support for graph-based ANN (HNSW via `ruvector-core`,
DiskANN via `ruvector-diskann`) and an IVF family (`ruvector-rairs`, ADR-193).
All existing graph indices build their neighbourhood structure from
**base-to-base** distances — an optimal strategy when queries and base vectors
share the same distribution.

### The OOD gap

Cross-modal retrieval violates this assumption.  In CLIP-style systems:

- **Base corpus**: image embeddings (distribution A).
- **Queries**: text embeddings (distribution B).

Distributions A and B occupy different geometric regions of the embedding space.
HNSW's greedy walk starts in the wrong region and fails to navigate to the true
nearest neighbours, producing recall@10 as low as 11.3% in our synthetic OOD
benchmark — versus 100% for brute-force.

This is the central problem addressed by RoarGraph.

### Existing approaches

| Approach | OOD strategy | Limitation |
|----------|-------------|-----------|
| HNSW (Malkov & Yashunin 2018) | None; base-to-base | Severe OOD recall degradation |
| DiskANN/Vamana (Subramanya et al. 2019) | Robust pruning | Still base-to-base |
| NSG (Fu et al. 2019) | Monotonic relative NN | Still base-to-base |
| RoarGraph (Chen et al. 2024) | Bipartite projection | Requires training queries |

## Decision

Introduce `crates/ruvector-roargraph` implementing the RoarGraph algorithm with
the following design:

### Core trait

```rust
pub trait AnnIndex {
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RoarError>;
    fn build(&mut self, training_queries: &[Vec<f32>]) -> Result<(), RoarError>;
    fn search(&self, query: &[f32], k: usize, ef: usize)
        -> Result<Vec<SearchResult>, RoarError>;
    fn len(&self) -> usize;
}
```

`BaselineGraph` ignores `training_queries`; `RoarGraphIndex` uses them for
the bipartite projection.

### RoarGraph build (bipartite projection)

```
For each training query q_i:
  knn = brute_force_knn(q_i, base_vectors, k_train)
  For each pair (u, v) in knn × knn, u ≠ v:
    projected_neighbours[u] ∪= {v}

For each base node u:
  sort projected_neighbours[u] by dist(u, neighbour)
  keep top max_degree neighbours

Connectivity pass (BFS from node 0):
  for each unreached node u:
    bridge = nearest reached node
    adj[u].push(bridge)     // u → bridge
    adj[bridge].push(u)     // bridge → u  (ensures BFS reachability)
```

The symmetric bridge insertion is essential — a directed-only bridge allows
greedy search to *leave* isolated node u but does not allow BFS *to reach* u
from the seed, which would violate the connectivity guarantee.

### Baseline (base-to-base k-NN graph)

`BaselineGraph` implements `AnnIndex` using exact O(N² · d) pairwise distance
computation, mirroring what HNSW's layer-0 achieves.  This is the "OOD-naive"
comparison point.

### Search (greedy beam)

Standard HNSW layer-0 search pattern:
- Max-heap of candidates ordered by (−distance, node_id).
- Max-heap of results ordered by (+distance, node_id), bounded to size `ef`.
- Termination: best candidate distance > worst result distance.
- Entry point: node 0 (production would use a pre-computed medoid).

### Workspace registration

`crates/ruvector-roargraph` is added to the workspace `members` list in
`Cargo.toml` immediately after `crates/ruvector-rairs`.

## Consequences

### Positive

- **Fills the OOD graph gap**: ruvector now has a first-class index for
  cross-modal retrieval workloads.
- **Dramatic recall improvement**: +88.8 pp recall@10 vs base-to-base baseline
  on the OOD synthetic benchmark (11.3% → 100.0%).
- **Lower search latency than baseline**: 18.5 µs vs 36.3 µs mean per query —
  a 2x speedup — because the projected graph places true neighbours closer in
  graph hops.
- **No unsafe code**: `#![forbid(unsafe_code)]` throughout.
- **No C/C++ dependencies**: pure Rust, WASM-compatible.
- **Composable**: the `AnnIndex` trait allows hot-swapping with RAIRS, DiskANN,
  or future quantised variants.

### Negative / Trade-offs

- **Requires a training query set**: if no query samples are available at build
  time, the bipartite projection degenerates.  In practice, any representative
  sample of the query distribution suffices.
- **Build is O(M · N · d)**: for M=500K training queries and N=10M base
  vectors, brute-force kNN at build time is infeasible.  Mitigation: use
  approximate kNN (IVF-routed) for the build inner loop.
- **Graph is not updatable**: adding new base vectors after build requires full
  or partial rebuild.  Standard limitation shared by HNSW.
- **Memory**: `O(N · max_degree · 4)` bytes for the adjacency list.  At
  N=5,000, max_degree=32: ~640 KB, well within DRAM limits.

### Neutral

- **Entry point is fixed at node 0**: production would precompute the corpus
  medoid.  For N=5,000 the fixed entry point has negligible impact on recall.
- **No SIMD**: the `l2sq` inner loop is scalar f32.  AVX2/NEON would give
  4-8x throughput at no algorithmic cost; orthogonal to this ADR.

## Benchmark Results (measured on 2026-05-18)

```
Hardware: Apple M4 Max, 128 GB RAM, macOS Darwin 24.6.0
Compiler: rustc 1.89.0 (29483883e 2025-08-04), --release
Dataset:  N=5,000 base (GMM-A, 8 clusters, dim=64, σ=0.5)
          M=500 train queries + 200 test queries (GMM-B, ood_shift=3.0)
          Ground truth = exact brute-force top-10
```

| Variant | recall@10 | mean latency | QPS | build time |
|---------|-----------|-------------|-----|-----------|
| Brute-force (exact) | 100.0% | 117.3 µs | 8,522 | — |
| Baseline (base-to-base k-NN) | 11.3% | 36.3 µs | 27,539 | 573 ms |
| **RoarGraph (bipartite projection)** | **100.0%** | **18.5 µs** | **54,088** | **256 ms** |

`cargo run --release -p ruvector-roargraph --bin roargraph-demo`

## Alternatives Considered

### 1. HNSW-style build using query vectors as extra nodes

Insert training queries as ephemeral nodes into the HNSW graph, then remove
them post-build, retaining their edges among base vectors.  Approximates the
bipartite projection without an explicit co-occurrence pass.  Rejected because
it requires integrating with the existing HNSW crate and the co-occurrence
approach is cleaner to implement, test, and reason about.

### 2. IVF routing for OOD queries

Assign OOD queries to IVF lists via an auxiliary cross-modal encoder and search
only those lists.  Requires a trained cross-modal mapping; not universally
available.  `ruvector-rairs` (ADR-193) covers the IVF family for same-distribution
queries; RoarGraph is the graph-index complement for OOD.

### 3. Approximate build kNN via RAIRS

Replace the O(M · N · d) brute-force kNN at build time with RAIRS search for
each training query.  Reduces build complexity to O(M · N^{1/2} · d) but
requires `ruvector-rairs` as a build-time dependency and complicates the crate
graph.  Tracked as future work.

### 4. ACORN-style filtered graph

ACORN (ADR-not-yet-written) supports predicate-filtered ANN by augmenting graph
edges with predicate annotations.  Could be combined with RoarGraph for filtered
cross-modal retrieval.  Not pursued here because filter integration is orthogonal
to the OOD graph construction algorithm.

## Related ADRs

- **ADR-143** (DiskANN / Vamana): disk-backed graph ANN; future disk-resident
  RoarGraph variant would reuse the memory-mapped storage conventions.
- **ADR-155** (RaBitQ+): 1-bit quantisation; integrating into RoarGraph's search
  inner loop would give 4-8x speedup in the candidate-scoring phase.
- **ADR-193** (RAIRS IVF): the IVF index family; could serve as the build-time
  approximate kNN oracle for large-scale RoarGraph construction.
