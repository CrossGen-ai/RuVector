# ADR-356: RCM reordering for HNSW layer-0 as an offline cache-locality pass

- **Status**: Proposed
- **Date**: 2026-11-07
- **Deciders**: ruv (owner), RuVector Architecture Team
- **Related**: `gist-ivf-centroid-locality-reorder-2026.md`, `gist-streamvbyte-hnsw-adjacency-2026.md`, ADR-353 (RVF stream descriptor)
- **Tags**: hnsw, locality, performance, cache, reorder

## Context

HNSW search on layer-0 is bandwidth-bound: each hop follows a pointer
into a neighbor whose vector may be anywhere in memory. Insertion-order
layouts put recently-added vectors near each other, which is weakly
correlated with graph proximity; after online churn the correlation
collapses to random. Measurements on a 50 k × 128-d small-world stand-in
show a greedy best-first search spends ~21 % more time on a randomly
permuted layout than on a bandwidth-reduced one, at identical recall and
identical visited count.

Reverse Cuthill-McKee (RCM) is a 50-year-old graph-bandwidth heuristic
from the sparse-matrix world. It is `O(|V| + |E|)` with a log factor for
per-level degree sort — for a 1 M × 32 HNSW layer that is a one-shot
~40 ms pass. It changes neither recall nor visited count; only where
vectors sit in memory.

This ADR proposes making RCM a first-class offline reorder strategy in
`ruvector-collections`, alongside the existing frequency-based reorder.

## Decision

Introduce an offline reorder strategy enum on HNSW indices:

```rust
pub enum LayerReorderStrategy {
    Identity,
    Frequency,          // existing: reorder by access frequency
    Rcm,                // new: reverse Cuthill-McKee on layer-0
    RcmThenFrequency,   // RCM for locality, then frequency shuffle within L1 lines
}
```

The PoC lives in `crates/ruvector-rcm-hnsw` and will be promoted into
`crates/ruvector-hnsw-freq-reorder` as a sibling function:
`rcm_reorder(adjacency) -> Vec<NodeId>`.

Per-collection reorder is user-triggered via a new CLI verb
`ruvector-cli hnsw reorder --strategy rcm`, and runs under a write lock.
No online / incremental variant ships in this ADR — hot-insert workloads
keep the existing layout.

## Consequences

**Positive**

- 9–21 % latency reduction on medium-large indices (50 k × 128-d) with no
  recall change.
- One-shot pass, easy to audit, deterministic.
- Composes with adjacency compression (streamvbyte deltas shrink when
  IDs are nearby).

**Negative**

- Index must be rebuilt on disk after reorder — not zero-copy. A
  background re-pack job must coordinate with reader snapshots.
- Online inserts after reorder drift the layout; benefit decays
  logarithmically in the mutation rate. Re-run RCM nightly for
  mutation-heavy indices.
- Negligible benefit on tiny indices that fit in L2; ADR explicitly scopes
  this to indices with `n × dim × 4 > L2_size`.

**Risk**

- Bugs in the BFS could emit a non-permutation and corrupt the index. The
  PoC asserts `|order| == n` and verifies uniqueness in tests; the
  production path will add a `debug_assert!` and also a `--verify` mode
  that recomputes recall on a fixed query set before committing the
  reorder.

## Alternatives considered

1. **Frequency-based reorder only** — already exists
   (`ruvector-hnsw-freq-reorder`). Works well when access patterns are
   skewed, but on uniform workloads it degenerates. RCM targets the
   structural graph, which is independent of workload.
2. **METIS / community detection** — gives tighter communities but costs
   `O(|V| log |V|)` or worse and requires a heavyweight dependency. RCM's
   linear cost + zero deps is the better default.
3. **Space-filling curve (Hilbert / Morton) over vectors** — assumes
   coordinate locality implies graph locality, which the HNSW build
   process already breaks via long-range edges. We rejected this early.
4. **Online incremental reordering** — valuable but orthogonal; tracked
   as future work (see research doc "What to improve next").

## Rollout

1. PoC crate shipped on `research/nightly/2026-11-07-rcm-hnsw` (this
   branch).
2. Promote `rcm_reorder` into `ruvector-hnsw-freq-reorder` behind feature
   flag `rcm`, default off.
3. Add CLI verb + bench harness on SIFT-1M.
4. Flip default to `RcmThenFrequency` once real-dataset wins replicate
   the stand-in numbers within 2 ×.
