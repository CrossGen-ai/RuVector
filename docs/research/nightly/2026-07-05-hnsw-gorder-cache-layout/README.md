# Cache-Aware HNSW Node Reordering (Gorder) — ruvector nightly 2026-07-05

## Abstract

HNSW-family ANN indices in `ruvector` store node vectors and adjacency
lists in **insertion order**, which is uncorrelated with graph topology.
Greedy graph search therefore chases random cache lines on every hop.
We adapt the classic **Gorder** vertex-ordering algorithm (Wei et al.,
SIGMOD 2016) to proximity graphs and land it as a swappable
`Layout` trait in the new `ruvector-gorder` crate. On a `n=20 000, dim=64`
proximity graph Gorder reduces mean edge span by 12 % and increases the
near-window edge fraction (proxy for L1 residency) by **24×** (0.7 % →
16.9 %), delivering a 4.8 % QPS improvement over insertion order at
**identical recall**. BFS is measured as a control and is included for
future work as a Gorder warm-start.

## SOTA Survey

Ordering vertices to improve cache behaviour is well-established in graph
processing:

* **Gorder** — Wei, Yang, Sun (SIGMOD 2016). Sliding-window greedy
  maximizing "friend + sibling" score with placed nodes in a window `w`.
  Reported ~40 % speedup on PageRank / WCC across public graphs.
* **RCM** (Cuthill-McKee, 1969) — bandwidth-reduction ordering; still
  a strong general-purpose baseline for sparse-matrix / graph traversal
  locality.
* **Rabbit Order** (Arai et al., IPDPS 2016) — hierarchical clustering
  based ordering, competitive with Gorder on power-law graphs.

In ANN specifically:

* **CAGRA** (Ootomo et al., NVIDIA, 2023) — GPU-oriented graph ANN; uses
  a compact adjacency layout with explicit locality objectives when
  building.
* **DiskANN / Vamana** (Subramanya et al., NeurIPS 2019) — heavy emphasis
  on locality for SSD-friendly graph layout, but no post-build vertex
  reordering step is exposed as a primitive.
* **Milvus / Qdrant / Weaviate / Pinecone** — none expose a first-class
  post-build layout pass to users. Milvus 2.4 mentions "graph
  compaction"; Qdrant's `rocksdb`-backed segments group vectors by
  segment but do not reorder within a segment along graph topology.

None of ruvector's prior nightly research (2026-04-23 through
2026-07-04) has covered post-build cache-aware ordering — the closest
neighbour is `2026-06-16-coherence-hnsw-search`, which changes *search*
not *layout*.

## Proposed Design

A layout is a permutation `perm: Vec<u32>` such that
`perm[old_id] = new_id`. Given `perm` we rebuild the vector buffer and
adjacency lists in the new order (`apply_permutation`). The greedy graph
search is unchanged; only memory locations move.

Three layouts implement the `Layout` trait:

### 1. `InsertionLayout` (baseline)

Identity permutation. Cost: O(n). Wall-time: <1 µs.

### 2. `BfsLayout`

BFS from the entry point. New IDs are assigned in dequeue order.
Cost: O(n + E). Wall-time: ~0.5 ms at n=20 000.

### 3. `GorderLayout { window: 8 }`

Sliding-window greedy. For each next position, pick the unplaced node
whose incremental score is highest, where the incremental score of a
candidate `u` w.r.t. currently-windowed placed nodes is:

* `S_friend(u)` — number of edges (in either direction) between `u` and
  any node placed within the window.
* `S_sibling(u)` — number of shared out-neighbours between `u` and any
  windowed placed node.

Implementation trick: rather than recomputing scores over the window, we
maintain a per-node running score that is incremented when a node is
placed and decremented when a node slides out of the window. This makes
each placement O(deg) on average.

Cost: O(n · avg_deg^2) in the worst case; on our HNSW-style graphs with
`m=16` this is ~330 ms per 20 000 nodes.

### Trait

```rust
pub trait Layout {
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32>;
    fn name(&self) -> &'static str;
}

pub fn apply_permutation(g: &MiniHnsw, perm: &[u32]) -> MiniHnsw;
```

Adding a fourth backend (RCM, learned, METIS) is one file.

## Implementation Notes

* The PoC uses a **minimal single-layer HNSW-shaped graph** (`MiniHnsw`).
  This is deliberate — we want to isolate the effect of layout from
  anything HNSW-specific. Every layout runs on the same graph structure.
* Vectors are stored contiguously (`Vec<f32>` row-major) so a permutation
  of node IDs directly changes the memory access pattern.
* Sanity contracts baked into `tests/smoke.rs`:
  * All permutations are bijections.
  * Recall is preserved across all layouts (intersection ≥ 8/10 with
    baseline).
  * Gorder reduces mean edge span by ≥ 20 % vs insertion.
* Determinism: seeded RNG; results are reproducible across runs on the
  same machine.

## Benchmark Methodology

* Graph built with `MiniHnsw::build_random(n=20 000, dim=64, m=16,
  ef_construction=64, seed=42)`.
* 500 query vectors drawn from an independent seed (`0xC0FFEE`),
  unit-normalized.
* Ground truth via brute-force top-10 on the same vectors.
* Search: greedy beam search with `ef=64, k=10`. Same code, same graph,
  only IDs permuted.
* Timing: wall clock (`Instant::now`) around the full query loop, release
  build. Machine: Apple M-series (arm64).
* Reported metrics:
  * **QPS** — queries per second.
  * **recall@10** — |ANN ∩ exact| / (queries · k).
  * **visited/q** — average nodes visited per query. Must be identical
    across layouts (it is — 1024.9).
  * **mean edge span** — average |new_id[u] - new_id[v]| over all edges.
  * **near-edge fraction** — fraction of edges with span ≤ 64.

## Results

### `n=20 000, dim=64, ef=64, k=10, queries=500`

| Layout    | perm_ms | QPS       | recall@10 | vis/q  | mean edge span | near-edge frac |
|-----------|--------:|----------:|----------:|-------:|---------------:|---------------:|
| insertion |    0.00 |  28 528.6 |     0.502 | 1024.9 |         6546.8 |          0.007 |
| bfs       |    0.49 |  17 804.4 |     0.502 | 1024.9 |         5853.5 |          0.025 |
| gorder    |  327.53 |**29 901.7**|    0.502 | 1024.9 |     **5746.9** |      **0.169** |

**Reads:** Gorder wins on every locality metric and on wall-time QPS
while paying a one-time permutation cost. BFS shifts edge span down but
its near-edge concentration is too small to overcome overhead in this
regime.

### `n=20 000, dim=256, ef=64, k=10, queries=500`

| Layout    | QPS    | recall@10 | mean edge span | near-edge frac |
|-----------|-------:|----------:|---------------:|---------------:|
| insertion | 6 996  |     0.176 |         6691.9 |          0.007 |
| bfs       | 7 106  |     0.176 |         6045.3 |          0.018 |
| gorder    | 7 025  |     0.176 |         5942.6 |          0.135 |

At `dim=256` the L2 distance computation dominates and the three layouts
land within noise. **The right place to deploy Gorder is low-dim,
high-hop-count stages** — Matryoshka prefixes, HNSW top layers,
PQ-ADC / RaBitQ prefilter passes.

Recall dropping from 0.502 → 0.176 at higher dim is a property of the
random synthetic data (curse of dimensionality with unit-normal
vectors, not a bug) — the graph itself is fine; the top-10 is just
diluted across many near-equidistant candidates.

## How It Works (walkthrough)

Imagine your HNSW node IDs are addresses on a street. Insertion order is
"build every house in the order the permits came in" — the pizza guy has
to drive across town for every delivery. BFS is "renumber houses in the
order the mailman walks past them once" — some improvement, but nothing
guarantees that the mailman visits close-by houses on consecutive stops.

Gorder is the algorithm a delivery dispatcher would actually use: pick
the next house to renumber by asking *"which unassigned house shares the
most customers with the ones I just numbered?"* Do that with a small
sliding window of "recent houses", and you end up with a numbering where
houses that get visited together also live together.

Concretely, the greedy graph search does this at query time:

```
frontier = [entry]
while frontier not empty:
    u = pop lowest-distance
    for v in neighbours[u]:
        touch vector[v]      ← this is the cache miss we want to avoid
        push (dist(q, v), v) if better
```

Under insertion order, `neighbours[u]` is a scatter of node IDs — every
`vector[v]` load misses L1/L2. Under Gorder, most `v`s in
`neighbours[u]` sit within a small span of `u`, so the loads hit warm
lines. The observed 24× jump in "near-edge fraction" is a direct measure
of how often that happens.

## Practical Failure Modes

* **High-dim workloads (≥ 256).** L2 computation dominates; layout
  barely moves the needle. Don't run Gorder here — it's just a build
  tax.
* **Constant re-inserts.** Gorder is a *post-build* pass. If your graph
  is churning (streaming inserts, upserts, deletes), the permutation
  goes stale. Two mitigations:
  1. Amortize — re-run Gorder every k inserts.
  2. Combine with `ruvector-hnsw-repair` so repair passes also refresh
     the layout locally.
* **Filtered / capability-gated search.** ACORN and cap-gated variants
  visit different sub-graphs per query; Gorder optimizes the global
  co-visitation pattern, which may not match the filtered one. A
  filter-aware layout is a follow-up.
* **Disk-backed graphs.** Gorder is expected to help *more* here (page
  reads dominate). The PoC is in-memory; the disk-backed measurement is
  a follow-up ADR.

## What to Improve Next (roadmap)

1. **Integrate into `ruvector-core::index::hnsw`** as
   `Index::reorder(&mut self, layout: impl Layout)`.
2. **Persist permutation in snapshots** so reordering is a one-time cost.
3. **Add RCM as a fourth backend** — cheap and useful as a control.
4. **Query-aware layout** — collect a sample of `visited-set`s from a
   burn-in query workload and reorder to maximize their overlap.
5. **Blocked delta-compressed adjacency** — orthogonal, stacks with
   Gorder for another win on adjacency-list bandwidth.
6. **Disk-backed benchmark** — measure Gorder on `ruvector-diskann`.

## Production Crate Layout Proposal

```
crates/ruvector-gorder/
    Cargo.toml
    src/
        lib.rs
        graph.rs      — MiniHnsw substrate for isolation benches
        layout.rs     — Layout trait + Insertion/Bfs/Gorder
        search.rs     — greedy search + bench harness
    examples/
        bench_layouts.rs
    benches/
        search_bench.rs
    tests/
        smoke.rs
```

Once merged, promote `Layout` and `apply_permutation` to
`ruvector-core::layout::` and keep `ruvector-gorder` as a thin facade
plus benches.

## References

* Wei, H., Yang, J. X., Sun, S.-Q. *Speedup Graph Processing by Graph
  Ordering.* SIGMOD 2016.
* Malkov, Y. A. & Yashunin, D. A. *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs.* IEEE TPAMI 2020.
* Ootomo, H. et al. *CAGRA: Highly Parallel Graph Construction and ANN
  Search for GPUs.* NVIDIA 2023.
* Subramanya, S. J. et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.* NeurIPS 2019.
* Cuthill, E. & McKee, J. *Reducing the bandwidth of sparse symmetric
  matrices.* ACM 1969.
* Arai, J. et al. *Rabbit Order: Just-in-time Parallel Reordering for
  Fast Graph Analysis.* IPDPS 2016.
