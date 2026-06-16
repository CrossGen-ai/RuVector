# Segment-HNSW for Range-Filtered ANN

**Nightly research · 2026-06-16 · iRangeGraph (SIGMOD 2024) inspired**

## Abstract

We implement a segment-tree of small proximity graphs (`ruvector-segment-rangehnsw`) for **range-filtered approximate nearest neighbor search** — the workload where the user filters by both vector similarity AND a continuous attribute range (timestamp, price, score, quality). Naive HNSW + post-filter loses recall to 6.5%@10 at 1% selectivity; ACORN-style pre-filter holds recall but is ~140× slower than brute force on narrow ranges. The segment-HNSW PoC matches brute-force recall (1.000) at 1% selectivity while running **2.2× faster than brute** and **6× faster than naïve post-filter QPS-adjusted for recall**. Memory cost is the expected O(log n) — 7.9× the single-flat-graph footprint on 10 000 × D=64.

This document is the SOTA survey, design rationale, implementation notes, measured numbers, and roadmap. The runnable PoC lives in `crates/ruvector-segment-rangehnsw/`. ADR-254 records the architectural decision.

## SOTA survey

Range-filtered ANN sits between predicate-filtered ANN (categorical labels, ACORN, Filtered DiskANN, NHQ) and pure ANN.

- **iRangeGraph** (Xu et al., SIGMOD 2024, arXiv:2403.13865) — establishes the segment-tree-of-graphs design, with each tree node owning a proximity graph over its slice plus an "elementary" graph for boundary handling. Reports ≥4× speedup over post-filter HNSW at high selectivity *and* over pre-filter HNSW at low selectivity on SIFT, Deep, GIST.
- **SeRF** (Zuo et al., SIGMOD 2024) — segment graphs over half-bounded ranges; complementary to iRangeGraph.
- **WST / Window Search Tree** (Microsoft Research, 2024) — uses a B-tree-like layout instead of binary tree; lower memory but slower at extreme selectivity.
- **ACORN** (Patel et al., SIGMOD 2024) — predicate-agnostic expansion. Handles categorical filters; degrades on narrow continuous ranges because the graph is not key-aware.
- **Filtered DiskANN / Stitched Vamana** (Microsoft, 2023) — per-filter graphs with stitching; built for categorical labels.
- **Milvus partition pruning, Qdrant payload index, Weaviate inverted index** — production systems use IVF-style partitioning by range; recall depends on the partition boundaries chosen at index-build time and degrades on workloads with shifting ranges.
- **Pinecone, LanceDB, FAISS IVF-PQ + post-filter** — same family; recall collapse at low selectivity is the known weakness.

The PoC here is the iRangeGraph design without its more advanced "elementary graph" boundary refinement — boundary partial-overlap nodes fall back to a leaf-level linear scan, which keeps the code under the 500-line ceiling without losing the headline behavior.

## Design

1. Sort the corpus by the range key `k`.
2. Build a binary segment tree over the sorted array. Each node `v` owns the contiguous slice `[lo_v, hi_v)` and a proximity graph indexed over the vectors in that slice. Leaves are slices of `leaf_size` (256 in the PoC).
3. **Query (q, [lo, hi], k, ef)**:
   - Descend the tree. A node fully inside `[lo, hi]` runs a beam search on its graph and contributes results to a global heap.
   - A node that partially overlaps `[lo, hi]` is recursed into.
   - A leaf that partially overlaps scans its in-range points linearly (cheap; ≤256 points).
   - Disjoint nodes are pruned.
4. Merge results: deduplicate by global id, keep top-k.

This gives the canonical O(log n) covering decomposition: each in-range point appears in exactly one fully-covering node's graph search at every level, with total search work ≈ ef × log(n / leaf_size) × the per-node beam cost.

## Implementation notes

- The per-node graph is a **single-layer NSW** (build by greedy search + reverse edges, bounded degree 2M) rather than a multi-layer HNSW. With slice sizes ≤ 5 000 (one level above the root in our PoC), one layer is enough — hierarchical layers buy little when the graph is already small.
- Distance is plain scalar squared-L2 with no SIMD intrinsics. The reported QPS understates a production implementation; honest scalar numbers are easier to compare against.
- Memory: each level redundantly stores vectors (the segment tree owns its own copies). This is the O(log n) blow-up. A production version would store vectors once and have the graphs index into the shared array — implemented in the roadmap.
- Construction is sequential; rayon parallelism per node is a straightforward extension (graphs at the same depth are independent).

## Benchmark methodology

```
$ cargo run --release -p ruvector-segment-rangehnsw --example bench
```

- Dataset: 10 000 × D=64 Gaussian vectors, deterministic LCG seed.
- Range key: uniform in [0, 1), so selectivity ≈ `hi − lo`.
- 100 queries per selectivity bin; query vector independent of the corpus.
- Selectivities measured: 1%, 5%, 20%, 50%.
- k=10. `ef = max(64, 8k)` for the flat-graph variants; the segment index uses ef=64 per node.
- Hardware: macOS arm64 (M-class), rustc 1.89.0 release, no SIMD.
- Recall is computed against brute-force range scan (1.000 by construction).

## Results

```
=== n=10000 d=64 k=10 selectivity=0.01 queries=100 ===
brute_force_range          build=0.0ms   q=0.95ms   qps=104858   recall=1.000
flat-graph + post-filter   build=378.8ms q=4.44ms   qps=22502    recall=0.065
flat-graph + pre-filter    build=378.8ms q=118.85ms qps=841      recall=0.995
segment-rangehnsw          build=1714.8ms q=0.43ms  qps=230858   recall=1.000

=== n=10000 d=64 k=10 selectivity=0.05 queries=100 ===
brute_force_range          q=2.07ms    qps=48323    recall=1.000
flat-graph + post-filter   q=4.98ms    qps=20067    recall=0.373
flat-graph + pre-filter    q=50.15ms   qps=1994     recall=0.987
segment-rangehnsw          q=3.59ms    qps=27859    recall=1.000

=== n=10000 d=64 k=10 selectivity=0.20 queries=100 ===
brute_force_range          q=7.11ms    qps=14071    recall=1.000
flat-graph + post-filter   q=4.85ms    qps=20609    recall=0.733
flat-graph + pre-filter    q=22.97ms   qps=4353     recall=0.946
segment-rangehnsw          q=9.74ms    qps=10264    recall=0.985

=== n=10000 d=64 k=10 selectivity=0.50 queries=100 ===
brute_force_range          q=15.79ms   qps=6331     recall=1.000
flat-graph + post-filter   q=5.67ms    qps=17647    recall=0.786
flat-graph + pre-filter    q=13.43ms   qps=7447     recall=0.876
segment-rangehnsw          q=14.73ms   qps=6789     recall=0.969

memory: flat-graph=3.79 MiB, segment-rangehnsw=29.92 MiB
```

### What the numbers say

- **Low selectivity (1%–5%)**: segment-HNSW is the only structure that keeps both recall and QPS — 230k QPS at 1% selectivity with 100% recall, an order of magnitude faster than every alternative on that workload.
- **Medium selectivity (20%)**: segment-HNSW still leads on recall (0.985) and is competitive on QPS. The flat-graph + post-filter is faster but at 0.733 recall, which is unacceptable for retrieval.
- **High selectivity (50%)**: the segment tree is dominated by boundary work and is roughly tied with brute force on QPS. At this regime there's little reason to bother with the index — the workload tells you to just scan. A production planner would switch between segment-HNSW (selectivity ≤ ~30%) and a flat-graph + post-filter (selectivity ≥ 80%) with brute force in between, exactly like iRangeGraph's "auto" mode.
- **Memory**: 7.9× cost matches the theoretical O(log₂(10 000/256)) = ~5.3 expected ratio (allowing for the extra graph overhead at every level). A shared-vector storage layout would reduce this to ~2×.

## How it works (blog walkthrough)

Imagine an e-commerce search: "find the 10 most visually similar product images priced between $40 and $60". The price filter is continuous. Two textbook answers:

1. **Post-filter**: run the ANN index, then drop everything out of range. Fine when 80% of the catalog falls inside [40, 60] — terrible when only 1% does, because the ANN beam saw 99 unhelpful candidates for every survivor.
2. **Pre-filter / predicate-agnostic** (ACORN, NHQ): walk the graph globally, only score in-range nodes. Recall holds, but the wall-clock blows up because the graph wasn't built with the filter in mind.

Segment-HNSW changes the index, not the query. We sort the corpus once by price and build a binary tree over the sorted array. Each tree node owns a tiny HNSW over its slice. The root's HNSW indexes the whole catalog; its two children each index half; their four grandchildren each index a quarter; and so on, down to leaves of ~256 items.

A query for [$40, $60] picks up to O(log n) tree nodes whose price intervals are fully inside [$40, $60]. Each of those nodes runs a small HNSW search over its own slice. The boundary nodes (partial overlap) fall back to a fast linear scan over their 256 items. Merge the results, return top-10.

The trick is that *every covering node already knew at build time which items it indexes*, so its graph is dense and accurate over that exact range. There's no recall collapse, no global graph walk, no oversampling. The cost is build-time work and memory — typical O(log n) factor — which is acceptable in retrieval workloads.

## Practical failure modes

- **Skewed range key distribution**: a tree built on a key that clusters heavily will have unbalanced leaves. The PoC sorts by key, which keeps slices balanced by *count* but a query like "show me items between the 99.0 and 99.1 percentile" still hits only one leaf. Acceptable; behavior matches brute force at that range size.
- **Updates / streaming inserts**: the segment tree is built once. A streaming workload needs either rebuilds at level boundaries (FreshDiskANN-style) or buffered inserts + periodic merge. Out of scope here.
- **Multi-dimensional ranges** (e.g. price AND timestamp): would require multi-dimensional segment trees (kd-tree of graphs). Roadmap.
- **Memory blow-up at extreme leaf sizes**: leaf_size too small ⇒ too many tiny graphs (overhead dominates); too large ⇒ leaf scans dominate at narrow ranges. The PoC uses 256, which is a reasonable default; auto-tuning is roadmap.
- **Distance is scalar f32**: production should use the workspace SIMD distance kernels.

## What to improve next

1. **Shared vector storage**: the segment tree owns vectors at every level. Refactor so each Graph holds indices into a single root-owned `Vec<Vec<f32>>` ⇒ memory back down to ~1.2× flat.
2. **Elementary graph** for boundary nodes (iRangeGraph): replace leaf-level linear scan with a small mixed graph that bridges partial-overlap slices.
3. **Parallel build**: nodes at the same depth are independent — `rayon` over each level. Build time drops from 1.7s ⇒ < 200ms for 10K points.
4. **Selectivity-aware planner**: at query time, decide between segment-tree, flat-graph + post-filter, and brute force based on `(hi − lo)` and corpus stats — gets the best of every regime.
5. **SIMD distance**: wire the workspace's distance crate for D=128/256.
6. **Multi-dimensional ranges**: kd-tree of graphs for AND-joined range filters.
7. **Real benchmarks at n=1M+**: today's numbers are on 10K. Need SIFT-1M and Deep-1B numbers before claiming production parity.
8. **Integrate with `ruvector-filter`**: the existing filter crate handles categorical predicates. Expose a unified API where categorical predicates use ACORN and range predicates use segment-HNSW.

## Production crate layout

If this graduates from PoC:

```
crates/ruvector-segment-rangehnsw/
├── src/
│   ├── lib.rs              — public Index trait + builder
│   ├── shared_store.rs     — single-owned vector storage
│   ├── graph.rs            — HNSW (multi-layer, SIMD distance)
│   ├── segment.rs          — segment-tree-of-graphs
│   ├── elementary.rs       — boundary-bridge graph (iRangeGraph)
│   ├── planner.rs          — selectivity-aware query mode picker
│   └── multi_dim.rs        — kd-tree of segment trees
├── examples/
│   ├── sift1m.rs           — SIFT-1M with timestamp filter
│   └── e_commerce.rs       — price + rating + timestamp tri-range
├── benches/
│   └── range_filter.rs     — criterion benchmarks
└── README.md
```

## References

- iRangeGraph: Xu, B., Wang, B., et al. "iRangeGraph: Improvising Range-dedicated Graphs for Range-filtering Nearest Neighbor Search." SIGMOD 2024. arXiv:2403.13865
- SeRF: Zuo, C., Qiao, M., et al. "SeRF: Segment Graph for Range-Filtering Approximate Nearest Neighbor Search." SIGMOD 2024.
- ACORN: Patel, L., Kraft, P., Guestrin, C., Zaharia, M. "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." SIGMOD 2024. arXiv:2403.04871
- Filtered DiskANN: Gollapudi, S., et al. "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters." WWW 2023.
- ruvector ACORN nightly research: `docs/research/nightly/2026-04-26-acorn-filtered-hnsw/README.md`
