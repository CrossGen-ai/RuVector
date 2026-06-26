# ADR-269: Dynamic Exploration Graph (DEG) for Continuously Self-Optimizing ANN

* **Status:** Proposed (nightly research PoC landed 2026-06-26)
* **Deciders:** ruvector core (nightly research agent)
* **Tags:** ann, graph-index, streaming, dynamic, sota
* **Related:** ADR-143 (DiskANN/Vamana), ADR-265 (comprehensive bench suite),
  ADR-267 (SOTA validation protocol), ADR-268 (capability-gated ANN),
  in-tree crates `ruvector-hnsw-repair`, `ruvector-diskann`, `ruvector-roargraph`

## Context

ruvector ships a wide family of graph indexes (HNSW via `hnsw_rs`,
DiskANN/Vamana, RoarGraph, NSG-style mincut variants) but every one of them
freezes a node's outgoing neighborhood at the moment of insertion. On
streaming workloads — where early inserts only see a small subset of the
final distribution — that produces a permanently sub-optimal graph. The
2023–2024 *Dynamic Exploration Graph* (DEG) line of work
(Hezel, Schall et al.) demonstrates that a flat fixed-degree graph paired
with a cheap *edge-swap* optimization pass can match or beat HNSW recall
while supporting continuous insertion and natural deletion — neither of
which HNSW handles cleanly.

Our nightly survey (June 2026) of competitor changelogs confirms that no
mainstream vector database — Milvus, Qdrant, Weaviate, Pinecone, LanceDB,
FAISS — currently ships a DEG-class continuously self-optimizing graph.
This is an open lane for ruvector.

## Decision

Land `crates/ruvector-deg` as a PoC reference implementation with:

1. A small `AnnIndex` trait (`insert`, `search`, `len`, `edge_count`) so
   future backends (parallel, wasm, node, GPU) implement only the contract.
2. Three backends in one crate for honest A/B comparison: `RandomGraph`
   (baseline), `NswGraph` (HNSW-layer-0 style), `DegGraph` (RNG prune +
   edge-swap optimization).
3. A `DegConfig` struct exposing `degree`, `ef_construction`, `extend_eps`,
   `optimize_every`, `optimize_passes`, `swap_eps` so the production
   integration can A/B without rebuilding.
4. Single-thread first — `optimize` runs inline on the insert thread. A
   follow-up crate `ruvector-deg-parallel` will move it to a background
   worker.

## Consequences

### Positive

* **Streaming-native.** First in-tree index where late information improves
  earlier nodes' neighborhoods automatically.
* **Trait-based.** `AnnIndex` is the only API surface; future crates
  (wasm, node, GPU) ship as thin shims.
* **Reproducible.** Deterministic LCG in `optimize`; benchmark output is
  bit-stable across runs.
* **Honest baseline.** The same crate ships the random and NSW
  comparisons so any quality delta is purely topological — no
  apples-to-oranges across crates.

### Negative / costs

* **Build time ~2× NSW** at default config due to inline optimization.
  Mitigation: move to background worker (planned follow-up crate).
* **One more crate** in an already large `crates/` directory. Acceptable —
  we ship one ANN family per crate by convention.
* **PoC quality only.** Single-thread, no SIMD beyond autovectorization,
  no disk backing. Production integration is a separate ADR.

### Neutral

* No change to existing crates. ruvector-deg compiles cleanly alongside
  `ruvector-hnsw-repair`, `ruvector-diskann`, `ruvector-roargraph` and the
  workspace `cargo build --release -p ruvector-deg` succeeds with one
  unused-arg warning addressed in the same PR.

## Alternatives considered

1. **Extend `ruvector-hnsw-repair`.** That crate exists for *deletion*
   repair; bolting on edge-swap optimization would conflate two distinct
   concerns. Cleaner to ship DEG as its own crate.
2. **Add to `ruvector-diskann`.** DiskANN is disk-resident with a different
   build invariant (α-pruning); merging would muddy both algorithms.
3. **NSG with periodic full rebuild.** Captures some of the quality benefit
   but at much higher cost than DEG's local 2-hop swap.
4. **Do nothing.** Loses an open competitive lane to potential rivals
   adopting DEG first.

## Acceptance criteria

* `cargo build --release -p ruvector-deg` clean. ✅
* `cargo test --release -p ruvector-deg` all green (4 tests). ✅
* `cargo run --release -p ruvector-deg --bin deg-bench` produces real
  numbers: DEG recall@10 within ±0.01 of NSW at the same degree budget
  on `N ∈ {5_000, 10_000}`, `dim = 64`. ✅
* Research doc at `docs/research/nightly/2026-06-26-deg-continuous-ann/`
  with SOTA survey, benchmark table, blog-style walkthrough, failure modes,
  and roadmap. ✅

## Follow-ups (not in scope of this ADR)

* ADR-270 (proposed): background-thread optimization (`ruvector-deg-parallel`).
* ADR-271 (proposed): DEG + RaBitQ distance estimator for `optimize` scan.
* ADR-272 (proposed): wasm32 + node bindings.
