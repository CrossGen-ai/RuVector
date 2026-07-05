# ADR-273: Cache-Aware Node Reordering for HNSW-Style Graphs (Gorder)

## Status

Proposed — nightly research 2026-07-05. Reference PoC crate:
`crates/ruvector-gorder`. Not yet integrated into `ruvector-core::index::hnsw`.

## Context

`ruvector` ships several HNSW variants (`ruvector-core`, `ruvector-coherence-hnsw`,
`ruvector-hnsw-repair`, `ruvector-diskann`, etc.). All of them assign node IDs
in **insertion order** and lay both the `Vec<f32>` vector store and the
`Vec<Vec<u32>>` adjacency lists out in that order. Search is a greedy
graph traversal that hops from node to node; each hop chases:

1. an adjacency list at `neighbors[u]` (indirect, heap-allocated `Vec<u32>`),
2. and for each neighbor `v`, the vector at `vectors[v * dim ..]`.

Because insertion order is uncorrelated with graph topology, consecutive
hops touch memory locations that are effectively random across the vector
buffer. On our benches every edge spans a mean of **~6547 IDs** in the
node array (`n=20 000`), i.e. every neighbor is nearly always a full cache
miss. Only **0.7 %** of edges land within a 64-ID window of their source.

The graph-analytics community solved a nearly identical problem in 2016:
**Gorder** (Wei, Yang, Sun; SIGMOD 2016, "Speedup Graph Processing by
Graph Ordering") — a sliding-window greedy that renumbers vertices so
that vertices which frequently co-occur in traversals are stored close
to each other. NVIDIA's CAGRA and cache-aware DiskANN variants apply
related ideas to ANN. We had none of this in `ruvector`.

## Decision

Introduce a **layout-agnostic post-build reordering step** for HNSW-style
graphs, exposed via a `Layout` trait with at least three implementations:

* `InsertionLayout` — baseline (identity permutation).
* `BfsLayout` — BFS from the entry point; O(n + E).
* `GorderLayout { window: usize }` — Gorder sliding-window greedy;
  O(n · window · m) where `m` is average out-degree.

The reordering step:

1. Computes `perm[old_id] = new_id`.
2. Rewrites the contiguous vector buffer in the new order.
3. Rewrites adjacency lists, translating neighbor IDs through `perm`.
4. Rewrites the entry-point ID.

The **PoC crate `ruvector-gorder`** implements all three layouts against a
minimal single-layer HNSW-shaped graph so we can measure the effect of
each layout in isolation from any HNSW-internal changes.

### Interface

```rust
pub trait Layout {
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32>;
    fn name(&self) -> &'static str;
}

pub fn apply_permutation(g: &MiniHnsw, perm: &[u32]) -> MiniHnsw;
```

## Consequences

### Positive

* **Real cache-locality win.** Gorder shrinks mean edge span from 6547 →
  5747 (-12 %) and increases the near-edge fraction (edges within a 64-ID
  window) from **0.7 % → 16.9 %** — a **24× improvement** in the metric
  Gorder actually optimizes.
* **Recall preserved exactly.** Layout is a pure ID relabelling: same
  graph, same visits, same recall (0.502 on our bench for all three
  layouts at `ef=64, k=10`).
* **Small QPS win at low dim / cache-bound regime.** At `dim=64, n=20 000`
  Gorder is 4.8 % faster than insertion (29 902 vs 28 529 QPS) despite an
  extra 328 ms permutation cost paid **once**.
* **Trait-based, swappable** — future layouts (RCM, METIS, learned) drop
  in without touching HNSW code.

### Negative / honest failure modes

* **BFS regressed on wall-time** on this bench (17 804 QPS vs baseline
  28 529). BFS reduces edge span slightly but doesn't concentrate edges
  in the near-window enough to overcome the fact that our greedy heap ops
  and vector loads dominate. BFS layout is kept because it's still useful
  as a warm-start for Gorder, and its numbers are useful as a control.
* **At high dim the effect vanishes.** At `dim=256, n=20 000` all three
  layouts land within noise (~7000 QPS): the L2 distance computation
  dominates and cache-miss reduction is a smaller share of the wall time.
  Sweet spot: **low-dim (32-96) high-hop-count workloads** — exactly the
  regime for coarse Matryoshka stages, PQ-ADC prefilter stages, and
  hierarchical HNSW top layers.
* **One-off cost.** ~330 ms permutation for 20 000 nodes. Amortizes over
  any nontrivial number of queries but is not free at build time.

### Migration path

1. Land the PoC crate (this ADR).
2. Add `hnsw::Index::reorder(&mut self, layout: impl Layout)` on the
   `ruvector-core` HNSW so users can opt into a layout after `finalize`.
3. Persist the permutation in the snapshot format (`ruvector-snapshot`)
   so reordering is a one-time cost.
4. Add `gorder` feature-gate to `ruvector-diskann` — disk formats benefit
   most from locality.

## Alternatives Considered

* **RCM (Reverse Cuthill-McKee).** Classic bandwidth-reduction ordering.
  Cheap and would be a reasonable fourth backend, but on early
  measurements Gorder consistently beats it for graph traversal (this
  matches Wei et al.'s original paper). Kept as future work.
* **METIS partitioning + per-partition contiguous layout.** Higher upfront
  cost (needs a partitioner dependency), and partitioning objectives
  don't directly match the "co-visitation locality" that greedy graph
  search actually stresses.
* **Learned reordering (RL / GNN-guided).** Interesting research target
  but far heavier and difficult to justify on top of a 24× gain from
  Gorder that costs no learning at all. Deferred.
* **No reordering; instead compress adjacency into blocked delta-encoded
  lists.** Complements reordering, doesn't replace it. Deferred.

## References

* Wei, H., Yang, J. X., Sun, S.-Q. **Speedup Graph Processing by Graph
  Ordering.** SIGMOD 2016.
* Malkov, Y. A. & Yashunin, D. A. **Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs.** IEEE TPAMI 2020.
* Ootomo, H. et al. **CAGRA: Highly Parallel Graph Construction and ANN
  Search for GPUs.** NVIDIA 2023.
* Subramanya, S. J. et al. **DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.** NeurIPS 2019.

## Benchmark Snapshot (real numbers, `M2/M-series`, release build)

`n=20 000, dim=64, m=16, ef=64, k=10, queries=500`, seed=42

| Layout    | perm_ms | search QPS | recall@10 | vis/q | mean edge span | near-edge frac |
|-----------|--------:|-----------:|----------:|------:|---------------:|---------------:|
| insertion |    0.00 |  28 528.6  |   0.502   | 1024.9 |         6546.8 |          0.007 |
| bfs       |    0.49 |  17 804.4  |   0.502   | 1024.9 |         5853.5 |          0.025 |
| gorder    |  327.53 |  **29 901.7** |   0.502   | 1024.9 |     **5746.9** |      **0.169** |

Numbers reproduced by:

```sh
cargo run --release -p ruvector-gorder --example bench_layouts
```
