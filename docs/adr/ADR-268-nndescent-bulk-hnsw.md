# ADR-268: NN-Descent Bulk kNN-Graph Construction for HNSW Base-Layer Bootstrap

**Status:** Proposed (PoC landed as nightly research crate `ruvector-nndescent`)
**Date:** 2026-06-22
**Related:** ADR-264 family (HNSW repair, Matryoshka), ADR-265 (benchmark suite), `ruvector-lsm-ann`, `ruvector-hnsw-repair`

## Context

Every HNSW-family index in RuVector (`ruvector-core`, `ruvector-coherence-hnsw`,
`ruvector-acorn`, `ruvector-matryoshka`, `ruvector-hnsw-repair`) is built today
by `N` sequential `insert()` calls. Cost is roughly `O(N · log N · ef_construction)`
distance computations, fundamentally serial, and dominates:

- agent cold-start time for `ruvector-agent-memory`,
- compaction time for `ruvector-lsm-ann`,
- `CREATE INDEX` time for `ruvector-postgres`,
- replication restore time for `ruvector-replication`,
- repair-after-delete cost for `ruvector-hnsw-repair`.

The literature has a well-established alternative: build the kNN graph *in bulk*
first, then use it as HNSW layer-0 adjacency. The bulk-build algorithm of choice
is **NN-Descent** (Dong, Charikar & Li, WWW 2011) and its derivatives
(EFANNA 2016, NSG 2019, Vamana / DiskANN 2019, CAGRA 2024). No RuVector crate
currently exposes a bulk kNN-graph builder.

## Decision

Land `crates/ruvector-nndescent` as a nightly research crate, scoped to:

1. A trait-based `KnnGraphBuilder` API as the long-term swap point for
   future bulk-build algorithms (NSG, Vamana, CAGRA-style reverse-kNN).
2. Two NN-Descent variants — *basic* (B[u]-only) and *local-join*
   (B[u] ∪ R[u], the canonical form) — plus a brute-force baseline.
3. A `SeededHnsw` that consumes a `KnnGraph` directly as layer-0
   adjacency and validates search-time equivalence via multi-entry-point
   beam search.
4. A reproducible benchmark binary (`cargo run --release --bin benchmark`)
   that reports build time, distance-op count, graph recall vs brute, and
   search QPS / recall for every variant.

Acceptance criteria, all met by the PoC (numbers in
`docs/research/nightly/2026-06-22-nndescent-bulk-hnsw/README.md`):

- `cargo build --release -p ruvector-nndescent` succeeds.
- `cargo test --release -p ruvector-nndescent` passes 4 real tests, including
  a `recall >= 0.95` and `distance-ops < 50%-of-brute` assertion for the
  local-join variant on N=1 200.
- At N=8 000, dim=64, K=16: local-join NN-Descent achieves **0.989 graph
  recall@16** in **13.9% of brute-force distance ops** and **1.84× faster
  wall time**.

## Consequences

**Positive**
- Establishes the first bulk kNN-graph construction primitive in the RuVector
  workspace. Future HNSW / Vamana / NSG / DiskANN-style crates have a shared
  build substrate.
- The trait-based design means SIMD / Rayon / Panorama-bound / GPU backends
  can land as drop-in replacements without changing downstream consumers.
- Concrete path to integrate with `ruvector-hnsw-repair` (SPFresh-style LIRE)
  and `ruvector-lsm-ann` (level compaction).

**Negative / cost**
- Adds another build-time algorithm surface. The team has to maintain it
  alongside the existing sequential HNSW build path.
- The PoC is single-threaded. Parallelization is mechanical but not yet
  written; until it lands, very large-N use cases still prefer threaded
  sequential insertion (hnswlib-style).
- `SeededHnsw` lacks upper-layer hierarchical routing, so its search-side
  numbers are a *navigability-equivalence* signal — not a competitive
  HNSW-search performance number. Wiring NN-Descent's output into
  `ruvector-core::HnswIndex` layer-0 is a follow-up.

## Alternatives considered

1. **Threaded sequential HNSW insertion (hnswlib model).** Already supported
   by some RuVector crates; gains 4-8× on hot core counts. Does not change
   the asymptotic distance-op budget. NN-Descent is complementary —
   threaded NN-Descent stacks on top.
2. **NSG / Vamana directly.** Both depend on a kNN graph as their starting
   point; building them first is putting the conclusion before the premise.
   We land NN-Descent first as the substrate, then add α-pruning /
   monotonic-edge refinement on top in a follow-up ADR.
3. **CAGRA (GPU NN-Descent).** Faster (~10×) but adds a CUDA dependency and
   excludes Apple Silicon / ARM-CPU / WASM targets. Out of scope for now;
   the CPU NN-Descent is the portable baseline they would replace.
4. **External crate (e.g. `instant-distance`).** No published Rust crate
   exposes a swappable NN-Descent backend; all known options are
   sequential-HNSW only. Building in-tree is the minimum lift to get the
   primitive into the workspace.

## Implementation summary

```
crates/ruvector-nndescent/
├── Cargo.toml
├── src/
│   ├── lib.rs            # re-exports
│   ├── distance.rs       # l2_sq + atomic DistanceCounter (~45 LOC)
│   ├── dataset.rs        # synthetic clustered data (~50 LOC)
│   ├── knn_graph.rs      # KnnGraph + KnnGraphBuilder trait (~75 LOC)
│   ├── brute.rs          # O(N²/2) baseline (~60 LOC)
│   ├── nndescent.rs      # basic + local-join NN-Descent (~230 LOC)
│   ├── hnsw_seeded.rs    # multi-entry beam search (~110 LOC)
│   └── bin/benchmark.rs  # benchmark + report (~130 LOC)
└── tests/integration.rs  # 4 acceptance tests (~80 LOC)
```

Every file is under 500 lines. All numbers in the research document
come from a real `cargo run --release` invocation on Apple M4 Max.
