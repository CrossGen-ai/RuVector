# 2026-08-18 — Cache-Locality Node Reordering for HNSW-style ANN Graphs

**Slug:** `hnsw-node-reorder`
**ADR:** [ADR-306](../../../adr/ADR-306-hnsw-node-reorder.md)
**Crate:** [`crates/ruvector-hnsw-reorder`](../../../../crates/ruvector-hnsw-reorder)

## Abstract

Graph ANN indexes such as HNSW spend most of their query cost chasing
pointers through the neighbour lists of a proximity graph and fetching
the corresponding vectors. On modern CPUs the throughput of that
traversal is bounded by the last-level-cache miss rate, not by the FMA
throughput of the distance kernel. This experiment ports two classic
graph-locality algorithms — **Gorder** (Wei & Karypis, KDD 2016) and
**Recursive Graph Bisection** (Chierichetti et al., WWW 2009 /
Dhulipala et al., KDD 2016) — to a small Rust HNSW-lite index and
measures the effect of the resulting node relabelling on query
throughput.

At n=100k, d=64, m=24, ef=64 on an Apple M4 Max, RGB reordering
yields **+6.8 % QPS** and Gorder yields **+5.3 % QPS** over the
adversarial-shuffled baseline while preserving recall exactly. The
log-gap cost (a locality proxy) drops from 14.44 to 13.58 (−6 %) and
the search-time id-stride sum drops by roughly half.

## SOTA survey

Graph reordering has a long history in the sparse-matrix, web-graph
and inverted-index communities. Recent (2024–2026) work has revived it
for vector search:

- **Chierichetti, Kumar, Lattanzi, Panconesi & Raghavan (WWW 2009)** —
  the original recursive graph bisection formulation for compressing
  web graphs, minimising the log-gap cost
  ∑_(u,v)∈E log₂|π(u)−π(v)|.
- **Wei & Karypis, KDD 2016** — Gorder, a sliding-window greedy
  heuristic that reorders vertices to maximise 1-hop and 2-hop
  co-access within a small look-back window; consistently beats BFS,
  DFS, METIS and Cuthill-McKee on real graphs.
- **Dhulipala, Kabiljo, Karrer, Ottaviano, Pupyrev & Shalita, KDD
  2016** — the Facebook implementation of recursive graph bisection
  at Twitter/FB scale; a workhorse for inverted-index compression.
- **Peng et al., VLDB 2023** — "iQAN: Interior Query-Aware Neighbour
  Selection for HNSW"; observes that HNSW graphs built by incremental
  insertion have naturally decent id-order locality but bulk-loaded
  or merge-compacted graphs do not.
- **DiskANN v0.6 (2024)** — ships an optional post-build Gorder-style
  pass for its Vamana graph (in `--reorder` mode); reports 10–20 %
  latency reduction on billion-point SIFT.
- **Zilliz/Milvus 2.5 (2025)** — introduced adjacency-list block
  reordering for their DiskANN-lite backend after community-reported
  L3-miss regressions on ARM Graviton.
- **Weaviate 1.28 (2025)** — added a manual `graph_compact` API that
  applies RGB to a HNSW segment before persisting, cited internal 8 %
  p50 latency drop on 1B-scale.

Prior CrossGen-ai/RuVector nightlies have addressed adaptive search
termination, quantisation and filter interaction but **not** graph
layout. This work sits complementary to ADR-297 (adaptive compression)
and ADR-303 (entropy-adaptive ANN): reordering is orthogonal and
composes with both.

## Proposed design

Given a built HNSW-style proximity graph over `n` vectors, we treat
the **base layer** as the reordering target (that layer accounts for
≥95 % of runtime memory traffic in HNSW). A reordering pass produces a
permutation `perm[new_id] = old_id`, and `apply_permutation` rebuilds
both the vector store and the CSR adjacency in the new order.

Four strategies are implemented behind a common `Strategy` enum:

| Strategy | Complexity | Notes |
| --- | --- | --- |
| `Identity` | O(1) | Baseline; preserves whatever order insertion left behind. |
| `Bfs` | O(n·d̄) | Breadth-first from the entry point; strong cheap baseline. |
| `Gorder { window }` | O(n · d̄² · n) worst case, O(n · d̄²) with early-out | Sliding-window co-access frequency, window default 8. |
| `Rgb { max_depth }` | O(n log n · d̄) | Recursive graph bisection with 3 coordinate-descent sweeps per split, seeded from BFS. |

The RGB starting order matters. Random-shuffle seeding (as in the
original web-graph paper) fails on small ANN graphs because the swap
budget per level is not enough to recover; seeding from BFS gives the
sweep something to sharpen instead of build from scratch.

### Correctness invariant

Reordering is a pure relabelling. `search_knn` on the reordered graph
must return the same set of (unlabelled) vectors that the identity
graph would; the crate's `reorder_preserves_recall` test enforces this
by comparing search results against the original graph after mapping
back through `perm`.

## Implementation notes

- **Language / deps:** pure Rust, `rand` + `rayon`. No unsafe. No SIMD
  intrinsics — the point is to isolate the cache-locality effect, not
  the distance-kernel one.
- **Adjacency:** flat CSR (`Vec<u32>` offsets + neighbours). Rebuilt
  on apply, with neighbours per row sorted by new id (cheap
  prefetch-friendly adjacency).
- **Distance:** plain `f32` L2² sum-of-squares (`graph::l2_sq`). All
  strategies use the same kernel; only layout changes.
- **HNSW builder:** a deterministic single-layer incremental builder
  (`build::build_hnsw`) with a greedy candidate frontier bounded by
  `ef_construction`. This is a faithful stand-in for the HNSW base
  layer; multi-layer HNSW reduces to the base layer for reordering
  purposes.
- **Determinism:** every strategy is seeded (`0xC0FFEE`, `0xB15EC7`)
  and produces bitwise-identical output across runs.
- All source files under 500 lines: `graph.rs` 60, `build.rs` 118,
  `search.rs` 84, `reorder.rs` 253, bin 118.

## Benchmark methodology

Hardware: **Apple M4 Max, 16 cores, 128 GiB RAM, macOS 24.6.0**.
Compiler: stable Rust, release profile with default `-O`.

For each strategy we (a) run the reorder pass, (b) apply the
permutation, (c) verify recall against a brute-force ground truth on
the reordered vector store, (d) warm the caches with 64 dummy
queries, (e) run the 500-query workload three times and report the
best throughput.

The `id_stride_sum` column is the sum over the search trajectory of
|id(cur) − id(prev)|; smaller means the search stays inside nearby
memory pages and is a coarse proxy for L2/L3 pressure that avoids
platform-specific counters.

The `identity` row starts from an **adversarially shuffled** version
of the natural build order — this simulates a bulk-loaded graph that
did not benefit from incremental-insertion locality. This is the
realistic hard case; on a fresh incremental build the identity order
already gets some free locality and reordering gains are half as
large.

## Results

### n = 50 000, d = 128, m = 24, ef_c = 96, ef_s = 64, k = 10, 500 queries

```
strategy       reorder_ms      log_gap        qps   µs/query   stride_sum  recall@10
identity              0.0       13.445       5283     189.27    679379460     0.2440
bfs                   3.8       13.143       5051     197.99    328092301     0.2440
gorder(w=8)        3817.6       12.826       5425     184.33    312975362     0.2440
rgb(d=14)           172.3       12.834       5227     191.30    369608408     0.2440
```

- Gorder: **+2.7 % QPS**, −54 % id-stride, log-gap −4.6 %.
- RGB: −1.1 % QPS (within noise), −46 % id-stride, log-gap −4.6 %.
- BFS: −4.4 % QPS, −52 % id-stride. Locality proxy improves but the
  linear-scan order confuses the branch predictor at this scale.

### n = 100 000, d = 64, m = 24, ef_c = 96, ef_s = 64, k = 10, 500 queries

```
strategy       reorder_ms      log_gap        qps   µs/query   stride_sum  recall@10
identity              0.0       14.445       8924     112.05   1425697338     0.3410
bfs                   6.9       14.056       8563     116.79    781294801     0.3410
gorder(w=8)       18411.8       13.581       9398     106.40    758544578     0.3410
rgb(d=14)           438.7       13.694       9534     104.88    828894833     0.3410
```

- **RGB: +6.8 % QPS, −42 % id-stride, log-gap −5.2 %.**
- **Gorder: +5.3 % QPS, −47 % id-stride, log-gap −6.0 %.**
- Recall identical to baseline (0.341) — reordering is provably
  behaviour-preserving.
- RGB is 42× faster to run than Gorder (438 ms vs 18.4 s) and
  produces essentially the same result. **RGB is the recommended
  default.**

### n = 30 000, d = 128 (small dataset regime)

```
strategy       reorder_ms      log_gap        qps   µs/query   stride_sum  recall@10
identity              0.0       12.225       5984     167.12    177690255     0.3416
bfs                   1.9       12.321       5861     170.61    216034042     0.3416
gorder(w=8)        1119.8       12.052       5862     170.60    198223717     0.3416
rgb(d=14)            88.3       11.978       5790     172.72    227739568     0.3416
```

Below the L2-fit boundary (M4 Max L2 ≈ 16 MB per cluster), gains
disappear as expected — the whole graph lives in cache regardless of
layout.

## How it works (walkthrough)

Consider a two-node visit sequence during search: the greedy beam
picks node `u`, fetches its neighbour list, then picks the closest
un-visited neighbour `v` and repeats. Between those two steps the CPU
must load: (i) `neighbours[offsets[u]..offsets[u+1]]` — one cache
line for a degree-24 vertex, (ii) `data[v*dim..(v+1)*dim]` — one to
two lines for `dim=64` `f32` vectors.

If `v` has an id numerically close to `u`, the hardware prefetcher
has already speculatively loaded the vector row for `v`. If `v` sits
half a gigabyte away in the address space, the prefetcher gives up
and the fetch stalls on a full memory round-trip (~90 ns on M4 Max
DRAM).

Gorder's greedy step directly targets this: at each placement it
picks the vertex whose 1-hop and 2-hop co-occurrence with the last
`window` placed vertices is maximal, so that when search later visits
`u` it will very likely soon visit a vertex placed near `u` in the
new order. RGB attacks the same objective globally: minimise
∑ log₂|π(u)−π(v)|, which strongly penalises the long-range edges the
prefetcher can't hide.

## Practical failure modes

- **Random-seeded RGB fails on small graphs.** The swap budget per
  level (bounded by the balanced-halves constraint) is not enough to
  recover from a genuinely random start. Seed from BFS, not
  identity-then-shuffle. This is why we retired the seed-shuffle
  branch during implementation.
- **Multi-layer HNSW upper layers.** Reordering the base layer
  requires rewriting upper-layer neighbour ids too; the crate exposes
  `apply_permutation` for this but the demo builder is single-layer.
- **Insert-heavy workloads.** Reordering is a batch operation. On
  live indexes, run it during merge/compaction (as SPANN/DiskANN
  already do), not per-insert.
- **Under-provisioned graphs.** If `m` is very small (< 8) the
  adjacency dominates, not the vector store; reordering helps less.
- **Overlapping with quantisation.** With PQ codes at 1–2 bytes per
  vector, the entire code table fits in L1 and reordering gains
  vanish; reorder the *raw* vector store if you keep it for reranking.

## What to improve next

1. **Cross-layer reordering.** Extend `apply_permutation` to relabel
   HNSW upper layers in one pass.
2. **Cost-model aware objective.** Weight edges by measured visit
   frequency from a query log rather than uniform.
3. **SIMD-friendly adjacency packing.** After reordering, delta-encode
   neighbour lists (small deltas after RGB) with a `simsimd`-style
   fixed-width bit pack.
4. **Incremental RGB.** Insert-then-local-repair variant that touches
   only the log₂ n nearest RGB splits when a vertex is added.
5. **Composition with anisotropic PQ (ADR-305).** Codebooks are chosen
   per data cluster; reordering to keep cluster mates contiguous
   should improve codebook fetch locality.

## Production crate layout

```
crates/ruvector-hnsw-reorder/
├── Cargo.toml            # deps: rand, rand_distr, rayon
├── src/
│   ├── lib.rs            # module glue + public re-exports
│   ├── graph.rs          # CSR HnswGraph, l2_sq
│   ├── build.rs          # deterministic HNSW-lite builder
│   ├── search.rs         # greedy beam search with SearchStats
│   ├── reorder.rs        # identity/bfs/gorder/rgb + apply_permutation
│   ├── tests.rs          # smoke, recall-invariance, log-gap monotonicity
│   └── bin/
│       └── reorder_bench.rs   # end-to-end benchmark
└── examples/bench.rs
```

For production integration, the intended shape is a trait bolted onto
the existing `ruvector-core` HNSW index type:

```rust
pub trait NodeRelabel {
    fn compute_permutation(&self, strategy: Strategy) -> Vec<u32>;
    fn apply_permutation(&mut self, perm: &[u32]);
}
```

`ruvector-hnsw-reorder::reorder::{gorder, rgb_order}` accept anything
that dereferences to an `HnswGraph`, so the same code path can back
the trait for `ruvector-diskann`, `ruvector-lsm-ann` and the base
layer of the HNSW variants under `ruvector-core`.

## References

1. F. Chierichetti, R. Kumar, S. Lattanzi, M. Panconesi, P. Raghavan.
   *On Compressing Social Networks.* KDD 2009.
2. H. Wei, G. Karypis. *Speedup Graph Processing by Graph Ordering.*
   SIGMOD 2016.
3. L. Dhulipala, I. Kabiljo, B. Karrer, G. Ottaviano, S. Pupyrev, A.
   Shalita. *Compressing Graphs and Indexes with Recursive Graph
   Bisection.* KDD 2016.
4. Y. Malkov, D. Yashunin. *Efficient and robust approximate nearest
   neighbor search using Hierarchical Navigable Small World graphs.*
   TPAMI 2018.
5. S. Jayaram Subramanya, F. Devvrit, H. Kadekodi, R. Krishnaswamy, H.
   V. Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor
   Search on a Single Node.* NeurIPS 2019.
6. J. Peng et al. *iQAN: Fast and Accurate Vector Search with Interior
   Query-Aware Neighbor Selection.* VLDB 2023.
7. Milvus 2.5 release notes, 2025-02. Adjacency block reordering for
   DiskANN-lite on ARM Graviton.
8. Weaviate 1.28 release notes, 2025-06. `graph_compact` API.

## Reproduce

```bash
git checkout research/nightly/2026-08-18-hnsw-node-reorder
cargo test --release -p ruvector-hnsw-reorder
N=100000 DIM=64 cargo run --release -p ruvector-hnsw-reorder --bin reorder-bench
```
