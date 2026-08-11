# ADR-299 — HNSW Neighbor Diversification via RNG / Vamana α-Prune

- **Status**: Proposed (nightly-research)
- **Date**: 2026-08-11
- **Slot**: nightly 2026-08-11
- **Related**: ADR-298 (centroid-seeded HNSW entry points), ADR-297 (adaptive
  compression retrieval plane)
- **Crate**: `crates/ruvector-hnsw-rng-diverse/`
- **Writeup**:
  [docs/research/nightly/2026-08-11-hnsw-rng-neighbor-diversification/README.md](../research/nightly/2026-08-11-hnsw-rng-neighbor-diversification/README.md)

## Context

Graph-based ANN indices (HNSW, Vamana/DiskANN, NSG) share a construction skeleton
and diverge chiefly in the **neighbor-selection heuristic** — how they choose which
of the ef_build candidates to keep as out-edges of a new node. The naive choice
(top-M nearest) produces highly connected, short-range graphs on which greedy
search dead-ends inside clusters. RNG-style diversification produces sparser but
navigable graphs and is what every SOTA graph ANN implementation actually ships.

Prior to this ADR, the `ruvector` graph indices used a top-M-style selector inherited
from the initial HNSW port. This ADR proposes migrating them to a `Pruner` trait
with RNG and α=1.2 Vamana implementations as the defaults.

## Decision

1. Introduce a `Pruner` trait in the graph module with three built-in variants:
   `Naive`, `RngPrune`, `AlphaPrune { alpha: f32 }`.
2. Default the flat and hierarchical graph builders to `AlphaPrune { alpha: 1.2 }`
   (matches DiskANN paper's recommended default and our measured sweet spot).
3. Expose the pruner as a build-time knob on `GraphIndexBuilder`.
4. Keep the `Naive` pruner available for regression / ablation runs only.

## Consequences

**Positive**
- On the reference nightly workload (N=1500, D=32, mixture, M=12, ef_build=40):
  - r@10 at ef_search=128 rises from 0.685 → 0.990 (+45 pp) with RNG.
  - r@10 at ef_search=128 rises to 0.985 with α=1.2 (+44 pp).
- Average out-degree drops 31 % under RNG (12.00 → 8.26), reducing graph storage.
- Query latency delta is negligible (~4 % more distance calls, ~10 % more µs/q).

**Negative**
- Build cost rises O(M²) per node in the worst case for the pairwise domination
  check; on typical M ≤ 32 this is dwarfed by ef-search distance computation.
- α is now a tunable; if set too high (≥1.5 in our sweep) the pruner degenerates
  toward naive and the recall gain disappears silently.

## Alternatives Considered

- **Keep naive top-M** — measured to be dominated on every recall metric at every
  ef_search; rejected.
- **Ship RNG (α=1) as default** — best raw recall but 31 % edge reduction is
  aggressive for users with unusual metrics; α=1.2 is a safer default that
  matches DiskANN.
- **NSG's MRNG constraint** — stricter than RNG; requires additional angular
  computation. Defer to a later ADR if sparsity is critical.

## Migration Plan

1. Land this crate as nightly research (`research/nightly/2026-08-11-*` branch,
   this ADR).
2. Follow-up PR: promote the `Pruner` trait into `crates/ruvector/`'s graph
   module, gated by a `pruner` config field defaulting to `alpha=1.2`.
3. Regression bench: run the naive vs α=1.2 sweep on the existing HNSW benches
   and confirm the ≥+30 pp r@10 lift at ef_search=128 persists.
4. Deprecate naive top-M in the next minor version.

## References

- Malkov & Yashunin, "HNSW," *TPAMI 2018* — <https://arxiv.org/abs/1603.09320>
- Subramanya et al., "DiskANN," *NeurIPS 2019*
- Fu et al., "NSG," *VLDB 2019* — <https://arxiv.org/abs/1707.00143>
- ADR-298 — companion nightly on entry-point selection (both are orthogonal).
