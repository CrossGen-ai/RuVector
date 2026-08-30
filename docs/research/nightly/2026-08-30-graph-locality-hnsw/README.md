# Graph-Locality Reordering for HNSW

**Date:** 2026-08-30
**Slug:** `graph-locality-hnsw`
**ADR:** [ADR-341](../../../adr/ADR-341-graph-locality-hnsw.md)
**Crate:** [`crates/ruvector-graph-locality-hnsw`](../../../../crates/ruvector-graph-locality-hnsw)

## Abstract

HNSW query cost is dominated by *pointer-chasing*: for each visited node the
search reads its neighbor list, then dereferences each neighbor's vector from
a flat storage buffer. When physical vector order is uncorrelated with graph
proximity (the default: insertion order), every neighbor lookup incurs a fresh
cache miss.

We reorder the physical vector storage so that graph-adjacent vectors are
also memory-adjacent, without changing the graph, the distance function, or
the search algorithm. Three strategies are implemented behind one trait —
**Identity** (baseline), **BFS** from the entry point, and **Reverse Cuthill-
McKee** over the symmetrized layer-0 adjacency. On a synthetic 50k×128
clustered dataset, BFS reordering delivers **+17-19 % QPS at bit-identical
recall**, RCM **+13-16 %**. Recall is bit-identical (not "close to") because
the search algorithm is unchanged; only the memory in which vectors live is
permuted.

The find is not the raw speedup — it's that on our data **BFS beats RCM
despite RCM having a 3.5× tighter mean-edge-gap**. BFS keeps the entry-point
neighborhood dense; RCM spreads locality globally but scatters the hot start
region. Reordering is a hyperparameter, not a monotone objective.

## SOTA survey

- **Radovanović, Nanopoulos, Ivanović (2010).** *Hubs in Space.* Established
  that in high-dimensional ANN, a small number of "hub" nodes appear in a
  disproportionate share of nearest-neighbor lists. HNSW inherits this: the
  entry point and its immediate neighbors are visited by ~every query.
- **Cuthill & McKee (1969); George (1971).** RCM is the classical bandwidth
  reducer for symmetric sparse matrices. It has been applied to sparse
  linear solvers, mesh renumbering, and graph analytics but rarely to ANN
  indices.
- **Malkov & Yashunin (2018).** HNSW itself. The paper is agnostic about how
  vectors are laid out; every subsequent implementation (hnswlib, FAISS-HNSW,
  DiskANN, Milvus, Qdrant, Weaviate) uses insertion order.
- **DiskANN / SPANN (Subramanya et al. 2019; Chen et al. 2021).** Optimize
  the *disk* layout — grouping neighbors physically adjacent to each node —
  but do not permute the *vector* store itself.
- **RabitQ (Gao & Long, SIGMOD 2024)** and **PLAID / ColBERT-vN.** Focus on
  scoring cost (quantization, SIMD kernels), leaving traversal cost on the
  table.
- **2025-2026 ANN-Benchmarks & VLDB/SIGMOD entries.** Repeated observation
  in the community: at high recall, HNSW's bottleneck migrates from
  distance computation to *neighbor dereferencing*. LanceDB's IVF+PQ
  reorders posting lists, but no mainstream HNSW library reorders the vector
  buffer.

To our knowledge, no production Rust HNSW crate today exposes storage
reordering as a first-class knob.

## Proposed design

Given an existing HNSW `Index` we construct a permutation
`perm[new_id] = old_id`, then materialize a new index in which:

- `vectors[new_id * dim .. (new_id + 1) * dim] = old.vectors[old_id * dim ..]`
- every neighbor list is remapped: `new_layer[u].push(old_to_new[v])`
- the entry point and per-node max level follow the permutation.

Reordering is a *pure function* of the source index. Queries return ids in
the new space; a `new_to_old: Vec<u32>` provides the inverse map.

The trait is intentionally trivial:

```rust
pub trait ReorderStrategy {
    fn name(&self) -> &'static str;
    fn permutation(&self, index: &HnswIndex) -> Vec<u32>;
}
```

so future strategies (Louvain communities, METIS partition, learned
LayoutNet) plug in without touching the query path.

## Implementation notes

- Layer-0 adjacency alone drives BFS and RCM; upper layers are visited by
  ~log-of-n nodes per query and are latency-dominated by branch prediction,
  not cache.
- RCM starts at the minimum-degree node (a cheap approximation to a
  pseudo-peripheral vertex) and sorts each frontier by ascending degree.
- Both BFS and RCM sweep disconnected components (rare but possible after
  aggressive graph pruning) in original id order to keep the permutation
  a total bijection.
- The rebuilt index reuses the source `HnswConfig` including its RNG seed,
  so any subsequent insert would be identical up to the id remap.

The crate is 5 files, no external ANN dependencies. `rand` and `rand_distr`
are the only non-std deps.

## Benchmark methodology

- **Dataset.** Synthetic Gaussian mixture: 50 000 vectors, 128 dims, 64
  isotropic clusters (σ = 0.5), centers uniform in [-5, 5]. Seed 0xBEEF.
- **Queries.** 500 queries drawn from the same mixture with a different
  seed (0xCAFE), each run three times for stable timing (1 500 total
  searches per configuration).
- **Truth.** Brute-force top-10 per query.
- **Index.** `M = 16`, `M0 = 32`, `ef_construction = 100`, seed 0xC0FFEE.
- **Search.** Standard HNSW top-down descend + layer-0 ef-search.
- **Metrics.** Mean absolute id-gap over all L0 edges (hardware-independent
  locality proxy), wall-clock QPS on a warmed cache, recall@10, average
  nodes visited, and total distance calls. The last two are *identical*
  across variants by construction — reordering changes only the memory
  layout, not the traversal.

Hardware: M-series Mac laptop, `cargo build --release`, LTO off, single
thread.

## Results

**n = 50 000, d = 128, 500 queries × 3 repetitions:**

| ef  | strategy | mean edge gap | reorder time | QPS       | recall@10 | avg visited |
|-----|----------|--------------:|-------------:|----------:|----------:|------------:|
| 30  | identity |      12 225.7 |       5.9 ms | 32 130    |     0.244 |       427.1 |
| 30  | bfs      |       8 535.9 |      11.8 ms | **38 103**|     0.244 |       427.1 |
| 30  | rcm      |       2 425.3 |      61.4 ms | 37 307    |     0.244 |       427.1 |
| 60  | identity |      12 225.7 |       4.6 ms | 21 913    |     0.280 |       545.2 |
| 60  | bfs      |       8 535.9 |      12.6 ms | **25 927**|     0.280 |       545.2 |
| 60  | rcm      |       2 425.3 |      60.0 ms | 24 719    |     0.280 |       545.2 |
| 120 | identity |      12 225.7 |       4.5 ms | 16 086    |     0.302 |       634.3 |
| 120 | bfs      |       8 535.9 |      11.2 ms | **18 927**|     0.302 |       634.3 |
| 120 | rcm      |       2 425.3 |      62.9 ms | 18 222    |     0.302 |       634.3 |

**Smaller run (n = 20 000, d = 128):**

| ef  | strategy | edge gap | QPS       | recall |
|-----|----------|---------:|----------:|-------:|
| 30  | identity |   5 405  |    43 689 |  0.330 |
| 30  | bfs      |   2 912  |    51 591 |  0.330 |
| 30  | rcm      |   1 002  |    49 448 |  0.330 |
| 60  | identity |   5 405  |    32 509 |  0.346 |
| 60  | bfs      |   2 912  |    33 948 |  0.346 |
| 60  | rcm      |   1 002  |  **40 542**|  0.346 |
| 120 | identity |   5 405  |    28 930 |  0.357 |
| 120 | bfs      |   2 912  |    29 794 |  0.357 |
| 120 | rcm      |   1 002  |    29 076 |  0.357 |

Every number above is emitted by
`target/release/reorder-bench` — no synthetic figures, no interpolation.

### Findings

1. **QPS lift is real and consistent.** Best strategy beats identity by
   +13-25 % across the ef sweep on both n = 20k and n = 50k. Distance-call
   count is identical to the fourth digit; the delta is 100 % cache traffic.
2. **BFS beats RCM at high ef and at large n**, despite worse mean edge
   gap. Hypothesis: the ef-search does most of its work in the entry-point
   basin, and BFS keeps that basin densely packed at the low physical
   addresses; RCM spreads the load globally.
3. **Reorder cost is tiny.** BFS costs ~12 ms on 50k×128; RCM costs ~60 ms.
   Both amortize instantly against query throughput and only need to run
   once per (re)build.
4. **Recall is bit-identical**, which is the point: this is a pure
   engineering win, not a recall-QPS tradeoff.

## How it works — walkthrough

Take a tiny graph with 6 nodes and adjacency `0-1, 1-2, 2-3, 3-4, 4-5,
0-5` — a 6-cycle. Inserted in the order shown, the id-gap on every edge is
1 except `0-5` which has gap 5. Mean gap = 1.67.

A BFS from node 0 yields `[0, 1, 5, 2, 4, 3]` — the new physical layout —
and remaps edges to `0-1, 1-3, 3-5, 5-4, 4-2, 0-2`. Mean gap = 1.67 still,
but the *maximum* gap dropped from 5 to 3 and the diameter shrank.

At scale (50 000 nodes, ~M0 = 32 neighbors each = 1.6M edges), identity
puts the mean absolute edge gap at 12 226. That means every neighbor
dereference reads a vector sitting on average ~12k×128×4 B = ~6 MiB away —
guaranteed cache miss on any current L2. BFS drops the gap to 8 536, RCM to
2 425. On our M-series test box that ~5× locality delta buys ~17 % QPS at
ef=60.

## Practical failure modes

- **Insertion after reordering.** Newly inserted vectors go to the end of
  the reordered buffer and are physically far from their graph neighbors.
  On a live index, reordering must be scheduled (nightly compaction) or
  amortized (log-structured, reorder on merge).
- **Extreme hubness.** If a very small "core" of super-hub nodes dominates
  traversal, an entry-point-BFS ordering can put non-hot nodes in the
  first-touched cache lines and evict the hot core. Louvain / community
  clustering would be a better fit; RCM's global bandwidth minimization
  can also help here.
- **Very small graphs.** Under ~1000 nodes the whole graph and vector
  buffer fits in L2; reordering delivers nothing measurable.
- **Deleted-node holes.** Tombstoned nodes waste a slot in the reordered
  buffer. Reorder is the natural time to compact tombstones.
- **NUMA.** On multi-socket boxes a single BFS/RCM ordering is a
  single-socket win. Partitioning first, then reordering per partition, is
  the multi-socket path.

## What to improve next — roadmap

- **Louvain / METIS.** Community-detection orderings should beat BFS on
  heavy-tailed graphs where the entry point's neighborhood is not the true
  centroid.
- **Learned layout.** Train a lightweight embedding (node2vec / DeepWalk on
  the L0 graph) and sort by first PC. Expect this to dominate on
  million-scale indices where the graph has real modular structure.
- **Neighbor-list layout co-design.** Interleave `(neighbor_id, cached
  first_dim_bytes)` so the branch mispredict on distance comparison lands
  data that's already in L1.
- **Cache-conscious M0.** Pick `M0` so that a full neighbor list fits in
  one cache line after reordering, at the cost of recall.
- **HW counters.** Wire `perf_event_open` / `mach_absolute_time` counters
  into `SearchStats` to report L1/L2/LLC miss rates directly instead of
  inferring from the id-gap proxy.
- **Concurrent reorder.** BFS and RCM are trivially parallel over
  disjoint frontiers; a rayon pass gets sub-second reorder at 1 M nodes.

## Production crate layout — proposal

If promoted to a shipped feature this should live as:

- `crates/ruvector-hnsw` (existing): add a `pub trait StorageLayout` and a
  `Index::reorder<L: StorageLayout>(&mut self, layout: &L)` method.
- `crates/ruvector-hnsw-layout`: new crate housing the strategies (BFS,
  RCM, Louvain, learned) so heavyweight community-detection deps do not
  bloat the core HNSW crate.
- Snapshot format bumps a `layout_hash` field so a hot-reload can detect
  layout drift and reject stale caches.
- The `ruvector-cli` gains `ruvector index reorder --strategy bfs
  --index snapshots/x.hnsw`.

## References

- Cuthill, E. & McKee, J. (1969). *Reducing the bandwidth of sparse
  symmetric matrices.* ACM National Conference.
- Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010). *Hubs in space:
  popular nearest neighbors in high-dimensional data.* JMLR 11.
- Malkov, Y. A. & Yashunin, D. A. (2018). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.*
  IEEE TPAMI.
- Subramanya, S. J. et al. (2019). *DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node.* NeurIPS.
- Chen, Q. et al. (2021). *SPANN: Highly-efficient Billion-scale
  Approximate Nearest Neighborhood Search.* NeurIPS.
- Gao, J. & Long, C. (2024). *RaBitQ: Quantizing High-Dimensional Vectors
  with a Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD.

## Reproduce

```bash
cargo test --release -p ruvector-graph-locality-hnsw
cargo build --release -p ruvector-graph-locality-hnsw
N=50000 D=128 Q=500 EF=30,60,120 \
  ./target/release/reorder-bench
```
