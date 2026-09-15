# ADR-346: Reverse-Neighbor Densification for HNSW Beam Search

- **Status**: Accepted (research prototype)
- **Date**: 2026-09-15
- **Crate**: `crates/ruvector-bidir-hnsw`
- **Author**: nightly-research agent

## Context

RuVector's production HNSW indices store a directed forward adjacency
`Adj[u] = {v₁ … v_M}` per node — the neighbors picked by
`select_neighbors_heuristic` at insertion time, plus reciprocal edges added
when a newly inserted node names an existing node as a neighbor.

In the steady-state graph, some nodes are pointed to *often* by others
(hub nodes / cluster medoids) but their outgoing lists — pruned by the
diversity heuristic — may miss the neighbors that actually point *to*
them. During query-time beam search we only follow the outgoing lists,
so those "reverse witnesses" are invisible. When `ef` is small, that
asymmetry costs recall: the beam settles into a local well and never
looks at nodes that are one reverse hop away.

## Decision

Introduce a `RevAdj` sidecar: after the HNSW graph is built, materialise
up to `r_max` in-neighbors per node into a compact flat table
(`n × r_max` × 8 bytes on 64-bit builds). Expose three search modes:

1. `Baseline` — vanilla forward-only beam search (unchanged).
2. `ExpandReverse` — at each step, expand candidates by
   `Adj[u] ∪ RevAdj[u]`.
3. `SymmetrizeBuild` — pre-symmetrise the forward adjacency (bounded by
   `m_max`, re-pruned by the diversity heuristic) so `Adj[u]` already
   contains the reverse witnesses.

`ExpandReverse` costs only the sidecar memory; the forward graph is
untouched, so existing snapshots and index files remain valid. It is a
strict addition — never removes candidates the baseline would visit.

## Consequences

**Measured on the shipped PoC** (n=20 000 mixture-of-Gaussians in
dim=64, m=16, ef_c=100, r_max=12, k=10, seed=42; single MacBook core, all
numbers from `cargo run --release -p ruvector-bidir-hnsw --bin bench`):

| ef  | baseline recall@10 | +expand-reverse | +symmetrize-build |
| --- | ------------------ | --------------- | ----------------- |
|  16 | 0.4450             | **0.7587**      | 0.4710            |
|  32 | 0.7597             | **0.9243**      | 0.7827            |
|  64 | 0.9317             | **0.9873**      | 0.9493            |
| 128 | 0.9937             | 0.9983          | 0.9940            |
| 256 | 0.9990             | 0.9993          | 0.9990            |

QPS at ef=64: baseline **8 128**, +expand-reverse **8 953** —
reverse expansion is *faster*, because it reaches the top-k neighborhood
sooner and terminates the beam earlier (avg dist evals drop from
731 → 665).

Extra memory: `r_max × 8` bytes per node — 96 B/node at r_max=12, so
≈1.9 MB for n=20 000, ≈95 MB for n=1 M. Cheap.

`SymmetrizeBuild` only recovers ~2 pp at low ef and matches baseline
above ef=64; the forward-only diversity heuristic is already close
to symmetric in practice, so the *sidecar* is what actually helps.

**Trade-offs**

- New memory cost proportional to `r_max`.
- Only useful in the low-ef regime; at ef ≥ ~128 the gains vanish
  because the baseline already saturates recall.
- The sidecar must be rebuilt on any structural mutation (insert /
  delete / repair). We recommend a lazy strategy: rebuild after every
  ≥5 % structural churn or on background compaction.

## Alternatives considered

1. **Pre-symmetrised forward adjacency (`SymmetrizeBuild`)** — measured
   above; barely moves the needle because HNSW's own reciprocation
   during construction already symmetrises most of the graph, and the
   *diversity re-prune* after adding back-edges tends to drop the very
   long-range links the reverse-expansion strategy relies on.

2. **Larger `m_max`** — a naive way to bring in more neighbors. Costs
   `(m_new − m_old) × 8` bytes per node in the forward table plus the
   quadratic diversity-heuristic cost during build. Empirically less
   efficient than the sidecar.

3. **RNG-diverse pruning refinements** (ADR-XXX-hnsw-rng-diverse, prior
   nightly) — orthogonal; the sidecar composes with any pruning
   policy since it only augments the query-time expansion.

## References

- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor
  Search Using Hierarchical Navigable Small World Graphs*, TPAMI 2020.
- Fu et al., *Fast Approximate Nearest Neighbor Search with the
  Navigating Spreading-out Graph* (NSG), VLDB 2019.
- Wang, Xu, Deng, *A Comprehensive Survey and Experimental Comparison
  of Graph-Based Approximate Nearest Neighbor Search*, VLDB 2021.
