# ADR-340: RoarGraph OOD-Aware Bipartite Augmentation for ANN

- **Status**: Proposed
- **Date**: 2026-08-29
- **Deciders**: RuVector nightly research
- **Related**: `crates/ruvector-roargraph`; docs/research/nightly/2026-08-29-roargraph-ood-bipartite/README.md; ADR-268 (ANN family); ADR-272 (recall-bounded ANN)
- **Tags**: ann, ood, cross-modal, rag, graph-index

## Context

Cross-modal RAG in ruvector — text queries searching image / audio /
video embeddings — hits a recall cliff that pure engineering (higher
`ef`, deeper `M`) does not close, because the failure is
distributional. Query vectors sit in a different region of embedding
space from base vectors, and a greedy walk over a graph built purely
from base-to-base neighbours routes them into the wrong region.

Chen et al. ("RoarGraph", VLDB 2024) address this by projecting a
sampled query workload into a bipartite graph and folding the resulting
base-side co-occurrence edges back into the base adjacency, along with
workload-aware entry points. Their paper reports up to 3.56× QPS at
fixed recall on Text-to-Image benchmarks over HNSW/NSG/DiskANN.

We built a compact, faithful-in-spirit implementation
(`ruvector-roargraph`) and measured it on a synthetic OOD dataset. It
delivers **3.04× recall@10** vs a base-only k-NN graph at fixed
`k=10, ef=48, dim=64, n_base=5000` for a **25% latency tax** and
**1.4% memory tax**. Numbers come from `cargo run --release` on Apple
M4 Max / rustc 1.89.0 with `seed=20260829`.

None of ruvector's existing 27 nightly ANN crates targets the OOD
routing problem directly. `ruvector-hybrid`, `ruvector-gnn-rerank`, and
`ruvector-matryoshka` are complementary but do not fix the graph.

## Decision

Adopt RoarGraph-style workload-projected augmentation as a first-class,
optional layer on top of any graph index in ruvector, exposed via a
new `AnnIndex` trait in `ruvector-roargraph`.

Concretely:

1. Publish `ruvector-roargraph` as a workspace member with three
   swappable backends (`FlatIndex`, `KnnGraphIndex`, `RoarGraphIndex`)
   behind a single trait.
2. Keep the augmentation algorithm decoupled from the base graph
   builder so a follow-up iteration can plug in `ruvector-hnsw-*`,
   NSG, or Vamana without touching augmentation code.
3. Enforce a bounded degree (`k_graph + k_aug`) on augmented nodes to
   keep serving-path latency predictable.
4. Ship the crate with `cargo test` covering exact recall of the flat
   backend, non-degenerate recall of the k-NN backend, and a
   structural guarantee that RoarGraph never regresses vs the base
   k-NN graph on OOD workloads by more than sampling noise.
5. Ship a `cargo run --release --bin benchmark` harness that produces
   real numbers on every invocation and prints both a human table and
   a machine-readable JSON block.

## Consequences

Positive:

- OOD RAG workloads (text-vs-image, code-vs-doc, query-vs-log) get a
  first-class recall lever inside ruvector.
- Trait-based design lets existing crates (`ruvector-hnsw-repair`,
  `ruvector-diskann`, `ruvector-spann`) contribute a base graph
  without refactoring.
- Honest benchmark + honest failure modes documented up front — future
  reviewers can trust the numbers instead of re-running from scratch.

Negative:

- Requires a representative query workload sample. Deployments without
  one gain nothing over `KnnGraphIndex`.
- Adds a rebuild cadence to production: workloads drift, and stale
  augmentation edges point to yesterday's OOD region.
- 25% mean-latency overhead is real. p95 latency budgets need to be
  audited before enabling by default.

Neutral:

- Extra memory is small in absolute terms (+1.4% at PoC scale, and
  sub-linear if the workload sample is bounded).

## Alternatives considered

1. **Query-side rotation / embedding remap** (RoBoost, 2025). Rotates
   the query embedding into the base distribution's basis. Cheaper at
   query time but requires a learned rotation per encoder pair;
   complementary to RoarGraph, not a substitute.
2. **Rerank with dense cross-encoder.** High quality, high cost, and
   dependent on a separate model. Belongs downstream of ANN, not
   inside it.
3. **Deeper `ef` on plain HNSW.** Cheapest to try; doesn't close the
   OOD gap because the graph itself routes wrong.
4. **Two indexes, one per modality, joined at query time.** Doubles
   memory and complicates writes; not competitive with a single
   augmented index.
5. **Do nothing / accept the recall cliff.** Currently the default
   ruvector behaviour on cross-modal workloads. Explicitly rejected —
   we now have a measured 3× lift on the table.

## Rollout

- Iteration 1 (this ADR): PoC crate lands on
  `research/nightly/2026-08-29-roargraph-ood-bipartite`, fork-only
  branch. No merge to `main`.
- Iteration 2 (future nightly): swap base graph to HNSW; benchmark on
  Text-to-Image-1B subset for external comparability.
- Iteration 3 (future nightly): streaming workload accumulation +
  graph-repair integration.
- Iteration 4 (future ADR): promote to `crates/ruvector-roargraph` in
  the production membership list with a feature-gated public API in
  `ruvector-core`.
