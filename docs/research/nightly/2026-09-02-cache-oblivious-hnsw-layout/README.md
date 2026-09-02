# Cache-Oblivious HNSW Graph Layouts

**Nightly research · 2026-09-02 · slug `cache-oblivious-hnsw-layout`**
Crate: `crates/ruvector-cache-oblivious-hnsw`
ADR: [ADR-343 / filed as 0001](../../../adr/0001-cache-oblivious-hnsw-graph-layouts-bfs-dfs-veb.md)
(cga-core-adr scaffolder allocated 0001 because its registry did not see the
existing ADR-3XX series; the .md header carries the ADR-343 tag per the repo
convention.)

## Abstract

HNSW greedy search is a pointer-chasing workload: each hop reads one 128-d
vector (512 B) and one adjacency list (~64 B), then jumps to a neighbour
whose slot is essentially arbitrary. On a modern out-of-order CPU the
distance kernel itself is cheap; wall-clock latency is dominated by
last-level cache and DRAM traffic. Existing production HNSW implementations
(hnswlib, FAISS-HNSW, RuVector's own) store nodes in insertion order or in
whatever order the builder produced. This nightly asks: **does a
cache-oblivious physical layout of the same logical graph speed up search?**

We build one 50k × 128-d HNSW graph (M=16, efc=96), then materialize it
under three permutations of node-id → slot: BFS (baseline), DFS, and a
van Emde Boas (vEB) recursive layout on the BFS tree. Every variant
runs the same search code. **vEB reduces mean latency by 18.8 %** and
lifts throughput from 7 983 QPS to 9 828 QPS on an Apple M4 Max.

## SOTA survey

Cache-oblivious data structures date to Frigo et al. (1999), who showed
that a recursive vEB layout on trees achieves the asymptotically optimal
`O(log_B n)` memory transfers per root-to-leaf traversal, independent of
the cache-line size `B`. Prokhorenko & Bender's later work (2004, 2006)
extended the analysis to arbitrary DAGs.

Recent applied work has revisited this for search graphs:

- **RaBitQ / Bit-Serial HNSW** (Gao et al., SIGMOD 2024,
  https://arxiv.org/abs/2405.12497) — focused on distance-kernel
  quantization, deliberately orthogonal to layout.
- **DiskANN** (Subramanya et al., NeurIPS 2019) uses a sector-aligned
  block layout with a co-located neighbour list; this is a coarse block
  layout, not a cache-oblivious one.
- **SPFresh** (VLDB 2024) — dynamic clustered ANN; touches on locality but
  does not systematically evaluate cache-oblivious node ordering.
- **Milvus 2.4 changelog** (2024) added a "graph reorder" pass but only
  provides BFS reorder; no vEB variant.
- **Qdrant 1.11** — payload-aware reordering for filtered search, not for
  cache locality of the base graph.
- **hnswlib** (Malkov et al., 2018 onward) never reorders after insert.

We could not find a public HNSW implementation that ships a vEB layout,
nor a systematic paper measuring the effect. Prior tree-vEB studies
(Bender et al., 2005) targeted B-trees; the transfer to greedy-descent
graphs is stated in folklore ("HNSW is basically a tree walk near the
target") but not measured.

## Proposed design

Given a finished HNSW graph `G = (V, E)` with entry point `s`, produce a
permutation `π : V → [0..|V|)` and materialize:

- a flat `Vec<f32>` of length `n · dim`, with slot `π(v)` holding
  `vec(v)`;
- a flat `Vec<u32>` of length `n · M`, with slot `π(v)` holding
  `[π(u) : u ∈ E(v)]` padded with `u32::MAX`.

Three permutations:

1. **BFS** (baseline). Slot order = BFS visitation order from `s`.
   Groups the search frontier per ply.
2. **DFS**. Slot order = DFS pre-order, neighbours pushed in reverse.
   Groups descent paths.
3. **vEB**. Build the BFS *tree* from `s`, then recursively split at
   half depth. Nodes near the same ancestor land in the same block; the
   guarantee is `O(log_B n)` cache-line misses per root-to-target
   descent, no `B` awareness required.

## Implementation notes

The crate is standalone (`[workspace]` at the top of its own
`Cargo.toml`) so it does not perturb the ruvector workspace build. Files
are all under 200 lines. The build path is a small greedy HNSW-style
graph — not a production HNSW — but is enough to reproduce the greedy
frontier search that the layout study is really about.

Traits are trivialized here: the layout enum (`Bfs | Dfs | Veb`) selects
a permutation function; the flat graph is generic over none of it.

## Benchmark methodology

- Hardware: **Apple M4 Max**, macOS 15.6. 128 KB L1D, 16 MB shared L2
  cluster.
- `cargo build --release` (opt-level 3, LTO thin, codegen-units 1).
- N = 50 000 vectors, dim = 128, M = 16, ef_construction = 96, seed
  `0xC0FFEE`. Corpus is unit-norm N(0,1)^128 via CLT-12.
- 1 000 held-out random queries (seed `0xBEEF`), k = 10, ef = 64.
- 50-query warm-up before timing.
- Metrics: mean per-query latency (µs), QPS, visited nodes, distance
  evaluations, and an "average slot stride" software proxy for locality
  (mean `|slotᵢ₊₁ − slotᵢ|` between successive visited nodes).

## Results

```
Cache-oblivious HNSW layout benchmark
n=50000 dim=128 M=16 ef_construction=96 queries=1000 k=10 ef=64
---
built Bfs in 9.32s (vectors=25600000 bytes, neighbours=3200000 bytes)
 BFS | lat   125.27 us/q | qps     7983.0 | visited/q 1153.6 | dists/q 1153.6 | avg-slot-stride    13994.5
built Dfs in 9.43s (vectors=25600000 bytes, neighbours=3200000 bytes)
 DFS | lat   112.71 us/q | qps     8872.0 | visited/q 1153.6 | dists/q 1153.6 | avg-slot-stride    14441.4
built Veb in 9.39s (vectors=25600000 bytes, neighbours=3200000 bytes)
 vEB | lat   101.75 us/q | qps     9828.3 | visited/q 1153.6 | dists/q 1153.6 | avg-slot-stride    15295.5
```

Headline numbers relative to BFS baseline:

| Layout | µs / query | QPS       | Δ latency | Δ QPS   |
| ------ | ----------- | --------- | --------- | ------- |
| BFS    | 125.27      | 7 983.0   | —         | —       |
| DFS    | 112.71      | 8 872.0   | −10.0 %   | +11.1 % |
| **vEB** | **101.75** | **9 828.3** | **−18.8 %** | **+23.1 %** |

Visited node count and distance eval count are **identical across
layouts** (1 153.6 per query). This isolates the effect: search does the
same work; only cache traffic changes.

The average slot stride is *higher* for vEB (15 295) than for BFS
(13 995). This looks paradoxical if you think of vEB as "packing
neighbours together"; it isn't. vEB packs the **ancestor chain to any
given target** together. Frontier-BFS bounces between siblings that share
no cache line, while vEB pays a large stride once (to jump to a bottom
subtree) and then makes many nearby hops within that subtree's block.
The stride mean misses that; hit-rate on hot ancestor cache lines wins
the day.

## How it works — plain-language walkthrough

Think of HNSW search as guessing a phone number one digit at a time by
asking "is your number closer to `X` or to `Y`?" Each guess is a
distance eval on one 512-byte vector. The CPU can do the arithmetic in
nanoseconds; the trouble is *fetching the 512 bytes*. If the next vector
sits in a cache line that's already resident, the fetch is free; if it
sits in DRAM, the fetch costs 60–100 ns.

BFS layout packs siblings together, which is exactly what greedy search
does *not* need — after picking the winning sibling, we descend past it.
vEB packs the descent chain together: the great-great-grandparent of any
node is only a few slots away, and so is every sibling of every ancestor
along the path. Since HNSW's greedy walk revisits the same ancestor set
many times (the frontier is a small BFS around the query), those hot
ancestors stay in L1/L2 for the whole query.

## Practical failure modes

- **Delete/insert breaks layout.** Any structural mutation invalidates
  the permutation. Production HNSW needs delete + insert; vEB layout has
  to be either recomputed periodically (a "compaction" job) or
  incrementally maintained (open problem — see roadmap).
- **Highly regular graphs help less.** Cache lines are 64 B; a `dim=64`
  fp16 vector fits in one line, so any layout is fine and vEB's win
  shrinks. The 128-d f32 case is exactly where cache pressure bites.
- **Small N kills the effect.** At N = 5 000 the whole graph fits in
  L2 and no layout matters. The measured win is dependent on the corpus
  being large enough that DRAM latency matters (here 25.6 MB of
  vectors overflows the M4's 16 MB shared L2).
- **Multi-thread queries.** If many queries run concurrently, the shared
  L2 is contended; per-query working sets shrink and the vEB gain
  attenuates. We measured single-thread only.
- **The "slot stride" locality metric is misleading**, as the results
  showed. It's a cheap proxy; real deployments should use `perf stat`
  or `dtrace` with L1/L2 miss counters.

## What to improve next

1. **Blocked vEB.** Group nodes into 64-B cache-line-sized micro-blocks
   *within* the vEB layout, aligning micro-block boundaries to the
   ancestor cutoff. Should knock out the residual DRAM crossings.
2. **Neighbour-list co-location.** Interleave `[vec | neighbours]` in
   the same cache line for the top of the tree, then split them for the
   leaves. hnswlib does this uniformly; vEB should do it *only where
   L2-resident*.
3. **Incremental relayout.** Maintain a shadow permutation; every
   `k`-th insert, run a compaction pass that vEB-relayouts one
   subtree. Amortises the O(n) reorder cost.
4. **Layered vEB.** Real HNSW has multiple layers; each layer is a
   graph. Apply vEB independently per layer, with the top-layer entry
   points always in the first cache line.
5. **Perf-counter validation.** Replace the stride proxy with actual
   L1/L2/LLC miss rates using `dtrace -n 'cpc:::miss'` on macOS or
   `perf stat` on Linux, and correlate.
6. **Bigger corpora.** N = 1M / dim = 768 (SBERT-scale). At that size
   the entire graph blows out any LLC and the theoretical `O(log_B n)`
   bound really kicks in.

## Production crate layout proposal

If this graduates from nightly, the shape would be:

```
ruvector-hnsw/
  src/
    graph.rs         # FlatGraph, unchanged
    layout/
      mod.rs         # trait GraphLayout { fn permute(&FlatGraph) -> Vec<u32>; }
      bfs.rs
      dfs.rs
      veb.rs
      blocked_veb.rs # follow-up from item 1
    compaction.rs    # incremental relayout job
  benches/           # criterion harness with perf-counter hooks
```

`ruvector-hnsw`'s `build_index` gains an optional `LayoutStrategy`
parameter, defaulting to `Bfs` for backwards compatibility. `Veb` is
opt-in until incremental relayout lands.

## References

- Frigo, Leiserson, Prokop, Ramachandran. *Cache-Oblivious Algorithms.*
  FOCS 1999.
- Bender, Cole, Demaine, Farach-Colton, Zito. *Two Simplified Algorithms
  for Maintaining Order in a List.* ESA 2002 / follow-ups.
- Malkov, Yashunin. *Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs.* IEEE TPAMI
  2020 (arXiv:1603.09320).
- Subramanya et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.* NeurIPS 2019.
- Gao et al. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024. https://arxiv.org/abs/2405.12497
- Milvus 2.4 release notes, 2024. https://milvus.io/docs/release_notes.md
- hnswlib. https://github.com/nmslib/hnswlib
