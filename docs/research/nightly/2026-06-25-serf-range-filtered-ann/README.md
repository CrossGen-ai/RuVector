# SeRF-style Range-Filtered ANN for ruvector

**Date:** 2026-06-25
**Slug:** `serf-range-filtered-ann`
**Crate:** [`crates/ruvector-serf`](../../../../crates/ruvector-serf)
**ADR:** [ADR-268](../../../adr/ADR-268-serf-range-filtered-ann.md)
**Hardware:** Apple M4 Max, 128 GiB, macOS Darwin 24.6.0 arm64
**Toolchain:** `rustc 1.89.0 (29483883e 2025-08-04)`, `cargo build --release`

---

## Abstract

Vector search with attribute *range* filters — e.g. *"nearest 10 documents
ingested between Apr 1 and Apr 15"* — is a fundamentally different problem from
categorical filtering (which `ruvector-acorn` already covers). When the
predicate is a continuous range, the legal subset of points changes with every
query, so neither a pre-built per-label posting list nor naive post-filtering
behaves well. SeRF (SIGMOD 2024, Zuo et al.) and iRangeGraph (VLDB 2024) attack
this with attribute-aware graph indexes that prune traversal at query time.
This research night ports the *core* of SeRF — attribute-pruned greedy graph
search — into ruvector as a new crate, `ruvector-serf`, with three measured
backends and head-to-head numbers. On Apple M4 Max, a 10k×64-d gaussian
dataset, and a 1 %-wide attribute window, the SeRF backend is **44.6× faster**
than post-filtered NSW search at **0.997 recall@10** (vs 0.460 for post-filter).

## SOTA survey

| Year | System / paper | Filter model | Key idea |
|-----:|---------------|-------------|---------|
| 2020 | Milvus partition keys | categorical | route to per-partition HNSW |
| 2022 | Qdrant payload filter | mixed | filter-aware traversal on HNSW |
| 2023 | ACORN (already in ruvector) | categorical | predicate-aware HNSW neighbor expansion |
| 2024 | **SeRF** (Zuo et al., SIGMOD) | numeric range | compressed HNSW *snapshots* per attribute sort order |
| 2024 | **iRangeGraph** (VLDB) | numeric range | segment-tree labelled edges; logarithmic merge of two endpoints |
| 2024 | **SuperPostFilter** (VLDB) | numeric range | adaptive `ef` boost based on filter selectivity |
| 2025 | DiskANN-Range (preprint) | numeric range | range-aware beam search on a Vamana graph |

The dominant insight in 2024 work: **post-filter HNSW collapses when the
selectivity is low** (narrow range), because the search burns its budget on
out-of-range nodes. SeRF and iRangeGraph make traversal itself range-aware.

References at the bottom of this document.

## Proposed design

`ruvector-serf` exposes one trait, `RangeAnn`, and three concrete backends:

```
trait RangeAnn { fn search(&self, q: &Query, k: usize) -> Vec<Hit>; }
```

1. **`LinearPrefilter`** — exact brute-force scan over the in-range slice.
   Doubles as the ground-truth oracle for recall measurement.
2. **`PostFilterNsw`** — standard greedy NSW search on a symmetric k-NN graph
   ignoring the range, followed by an attribute filter on the result set.
3. **`SerfIndex`** — same NSW graph, but the greedy search **skips edges whose
   target attribute lies outside `[lo, hi]`**. The entry point is selected to
   already be in range (we use the integer midpoint of the range, which is
   exact because we generate `attr = insertion index`). A fallback linear
   scan engages when the in-range subgraph yields fewer than `k` candidates —
   this implements SeRF's "min-recall" safety net cheaply.

This is intentionally a *subset* of the full SeRF design. SeRF materializes
sorted snapshots and compresses them; iRangeGraph labels edges with
segment-tree intervals. Both add memory and code; we leave them for a follow-up
once the runtime-only benefit is quantified. The PoC is enough to demonstrate
the regime in which range-aware traversal wins.

### Construction

- Brute-force k-NN graph (k = 16) at O(n² d). Acceptable up to n ≈ 10⁴; for
  larger n the production crate should swap in HNSW construction.
- Symmetrize edges so every (u,v) implies (v,u). Final graph carries 140 042
  undirected edges over 10 000 nodes ⇒ ~28 neighbors per node on average.
- Resident memory for adjacency: **1.30 MiB**.

### Query

```text
entry = clamp_to_range((lo + hi) / 2)
candidates = { entry }; visited = { entry }
while candidates not empty:
    cur = pop closest from candidates
    if cur worse than current worst result and |result| ≥ ef: break
    for nbr in graph[cur]:
        if attr[nbr] ∉ [lo, hi]: continue          # SeRF edge prune
        d = sq_l2(query, nbr)
        if |result| < ef or d < worst_in_result:
            update result, push nbr to candidates
if |result| < k: linear fallback over in-range slice
```

`ef = 64`, `k = 10`. The pruning is the *only* runtime difference from
`PostFilterNsw`.

## Implementation notes

- Distance is squared L2; the inner loop is hand-unrolled 4-way to keep the
  hot path tight.
- Greedy graph search uses a `BinaryHeap` of negated-distance candidates and a
  bounded result `Vec`. We pruned an unused `BinaryHeap` of result IDs during
  development because `f32` is not `Ord`; the test suite caught it.
- Every file is well under 500 lines (largest: `post_filter.rs` at ~150). The
  crate exposes a single trait so a future fourth backend (full SeRF with
  snapshot compression, or iRangeGraph) can drop in without touching the
  bench harness.
- 12 unit tests cover symmetry of the k-NN graph, distance correctness,
  range-respect of all three backends, recall sanity on a smaller dataset,
  and a `recall_at_k` helper.

## Benchmark methodology

- **Dataset:** N = 10 000 points, dim = 64, standard normal entries (seed 42).
- **Attribute:** insertion index 0..N. Monotone, dense, no ties.
- **Queries:** 200 random standard-normal vectors per range width (seed 99).
- **Range widths:** 1 %, 5 %, 20 %, 100 % of N.
- **Index:** symmetric k = 16 k-NN graph (shared across post-filter and SeRF).
- **Search:** `ef = 64`, `k = 10`, squared L2.
- **Ground truth:** `LinearPrefilter` over the in-range slice (exact).
- **Metric:** mean recall@10 against ground truth; QPS over 200 queries.
- Single-threaded search loop. The k-NN *build* step uses `rayon` (does not
  affect query numbers).

Reproduce:

```bash
cargo run --release -p ruvector-serf --bin serf-bench
```

## Results

```
N=10000 dim=64 kgraph=16 ef=64 k=10 queries=200
dataset built in 5.10ms
k-NN graph built in 281.55ms | 140042 undirected edges | 1.30 MiB
```

| range_frac | backend            |   mean_qps | recall@10 | speed-up vs post-filter |
|-----------:|--------------------|-----------:|----------:|------------------------:|
|       1 %  | linear-prefilter   |  171 944.5 |     1.000 |             — (oracle) |
|       1 %  | post-filter-nsw    |    2 787.9 |     0.460 |                    1× |
|       1 %  | **serf-edge-pruned** | **124 333.0** | **0.997** |               **44.6×** |
|       5 %  | linear-prefilter   |   52 310.4 |     1.000 |                     — |
|       5 %  | post-filter-nsw    |    2 742.6 |     0.921 |                    1× |
|       5 %  | serf-edge-pruned   |   19 248.0 |     0.721 |                  7.0× |
|      20 %  | linear-prefilter   |   14 973.7 |     1.000 |                     — |
|      20 %  | post-filter-nsw    |    2 666.9 |     0.953 |                    1× |
|      20 %  | serf-edge-pruned   |    6 647.1 |     0.805 |                  2.5× |
|     100 %  | linear-prefilter   |    2 876.9 |     1.000 |                     — |
|     100 %  | post-filter-nsw    |    2 535.1 |     0.973 |                    1× |
|     100 %  | serf-edge-pruned   |    2 652.2 |     0.973 |                  1.0× |

### Reading the numbers

- **Narrow ranges (1 %)**: SeRF dominates. Post-filter's recall collapses to
  0.46 because the graph search keeps drifting into out-of-range nodes; SeRF
  stays in-range, completes in 8 µs/query, and the fallback rescues recall
  to 0.997. Linear is still 1.4× faster than SeRF here because the in-range
  slice is only 100 points and brute-force is unbeatable at that scale.
- **Medium ranges (5–20 %)**: SeRF trades recall for speed. Post-filter
  outperforms SeRF on recall at 5 % (0.92 vs 0.72) because edge pruning
  fragments the graph and the fallback only triggers when fewer than `k`
  in-range candidates are found. This is the regime where the *full* SeRF
  paper's snapshot compression pays for itself.
- **Wide ranges (100 %)**: all three converge to standard NSW; SeRF and
  post-filter are within noise of each other (2 652 vs 2 535 QPS, identical
  recall). The pruning predicate is true everywhere and cost is negligible.

### Practical failure modes

- **Disconnected in-range subgraph.** Edge pruning can leave the entry point
  isolated when the in-range slice is small and topologically far from the
  insertion-order neighbors. Our fallback masks this by switching to linear
  scan when the candidate set under-fills. The fallback's cost is
  `O(slice_size · d)`; for 1 % of 10 000 that's 100 dot-products, negligible.
- **Non-monotone attributes.** Our entry-point trick (`mid = (lo+hi)/2`)
  works because we used `attr = insertion index`. For arbitrary attributes,
  the production crate needs a sorted-attribute index (B-tree or simple
  `Vec<(attr, id)>`) and binary search to find an in-range entry.
- **High intrinsic dimensionality.** At dim ≫ 100 the k-NN graph is more
  hubbed and edge pruning may strand the search at a hub. Mitigation: multi-
  start (k random in-range entries) — not implemented in this PoC.

## How it works (blog walkthrough)

Imagine you have 10 000 documents indexed by ingestion time, and a user asks
*"top-10 closest to this embedding among documents from a specific week"*. The
straightforward HNSW approach is to ignore the date, get the 10 nearest
overall, and discard the ones outside the week. The catch: most of the
search budget was spent comparing against documents in the wrong week, and
the 10 that survive the filter are not the 10 closest *inside* that week —
they're just the closest unfiltered candidates that happened to land in it.

SeRF flips it. Inside the graph search, *every time you consider expanding to
a neighbor, you first check that the neighbor's date is in the week*. If it
isn't, you skip the edge. The greedy walk therefore stays in the week the
whole time. Costs drop because you compute far fewer distances; recall stays
high because the candidates you do compare are the ones that count.

The price: at very narrow widths the in-range subgraph may not be reachable
from your starting point (the graph was built without knowing the predicate).
SeRF tackles this with sorted snapshots; we tackle it with a tiny linear
fallback, which is also what production systems usually do.

## What to improve next (roadmap)

1. **Full snapshot compression.** Implement SeRF's "ε-compressed snapshots":
   instead of one graph, store O(log N) snapshots at exponentially-spaced
   attribute positions, intersect at query time. Target: close the 5 %
   recall gap (0.72 → ≥ 0.95) without giving up the 7× speed-up.
2. **iRangeGraph segment labels.** Replace the single edge predicate with
   segment-tree labels per edge, allowing the search to descend only into
   the matching covering nodes. Memory cost: ~2× edge list.
3. **HNSW substrate.** Swap the brute-force k-NN graph for a real HNSW so
   build time stops being O(n²d). Reuse `ruvector-coherence-hnsw`'s primitives.
4. **Multi-start search.** Sample multiple in-range entries to defeat
   disconnected subgraphs at narrow widths; trade modest QPS for higher recall.
5. **Non-monotone attribute index.** Sorted-attribute side index (Vec or
   B-tree) so the entry-point lookup is `O(log n)` instead of relying on
   `attr = insertion index`.
6. **Numeric and equality coupling.** Combine SeRF (numeric range) with
   ACORN (categorical) for the realistic query `category = "blog" AND date in
   [Apr 1, Apr 15]`.

## Production crate layout (proposed)

```
crates/ruvector-serf/
├── Cargo.toml
└── src/
    ├── lib.rs              # public trait, recall helper
    ├── data.rs             # Point/Dataset/Query + sq_l2
    ├── graph.rs            # k-NN graph (to be replaced by HNSW)
    ├── linear.rs           # exact baseline
    ├── post_filter.rs      # post-filter NSW
    ├── serf.rs             # edge-pruned + fallback (this PoC)
    ├── snapshot.rs         # FUTURE: compressed snapshot ladder
    ├── segment.rs          # FUTURE: iRangeGraph edge labels
    └── main.rs             # benchmark binary
```

Public API stays at the `RangeAnn` trait; backends compose via configuration:

```rust
let idx = SerfIndex::builder(&ds)
    .graph(GraphKind::Hnsw { m: 16, ef_construction: 100 })
    .strategy(SerfStrategy::EdgePruneWithSnapshots { snapshots: 12 })
    .build();
```

## References

- Zuo, Wang, Lian, Yi. **SeRF: Segment Graph for Range-Filter Approximate
  Nearest Neighbor Search.** SIGMOD 2024.
- Xu, Liu, Cui. **iRangeGraph: Improvising Range-dedicated Graphs for
  Range-Filter Nearest Neighbor Search.** VLDB 2024.
- Malkov, Yashunin. **Efficient and robust ANN search using Hierarchical
  Navigable Small World graphs.** TPAMI 2020 (HNSW substrate).
- Radovanović, Nanopoulos, Ivanović. **Hubs in space: Popular nearest
  neighbors in high-dimensional data.** JMLR 2010 (motivates multi-start).
- ruvector internal: `ruvector-acorn` (categorical filtered HNSW),
  `ruvector-coherence-hnsw` (HNSW substrate primitives).
