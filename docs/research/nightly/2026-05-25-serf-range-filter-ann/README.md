# SeRF — Segment-Graph Range-Filtered ANN for ruvector

**Date:** 2026-05-25  **Crate:** `ruvector-serf`  **ADR:** ADR-194

## Abstract

Range-filtered approximate nearest-neighbor search ("k-NN among items with key
in `[lo, hi]`") is the gating workload behind real applications of vector
search: "find the most semantically similar log lines from the last 24 h",
"nearest products with price in $50–$120", "RAG over documents from this
quarter only". Naïve options are bad in opposite ways: pre-filter (linear scan
over a subset) collapses for low-selectivity ranges; post-filter (search the
global graph, then drop out-of-range hits) collapses for high-selectivity
ranges because the graph happily returns nearest-overall, not nearest-in-range.

This nightly delivers `ruvector-serf`, a Rust implementation of the
**segment-graph** family of range-filtered ANN (Zuo & Deng, VLDB 2024 SeRF;
Xu et al., SIGMOD 2024 iRangeGraph). A segment tree over rank-space holds one
NSW graph per canonical node; a range query is decomposed into O(log n) such
nodes and each is searched independently. We benchmark it head-to-head with
brute-force post-filter and a single-graph post-filter at three selectivities.

**Headline measured numbers (release build, M-class Apple Silicon,
N=20 000, D=128, k=10, NQ=200 queries):**

| selectivity | flat-postfilter (truth) | nsw-postfilter           | serf-segment-graph     |
|------------:|------------------------:|--------------------------|------------------------|
|        100% | 1.000  /  718 µs/q      | 0.517  /  147 µs/q       | **0.748**  /  419 µs/q |
|         10% | 1.000  /  100 µs/q      | 0.410  /  137 µs/q       | **0.859**  /  158 µs/q |
|          1% | 1.000  /  17.5 µs/q     | 0.085  /  155 µs/q       | **1.000**  /  8.5 µs/q |

At 1% selectivity, the segment graph is **18× faster than the single-graph
post-filter and 12× more accurate** at the same time. At 10% selectivity,
recall jumps from 0.41 to 0.86 while latency stays comparable. The 100% case
is the "any global graph wins" regime — and even there segment-graph beats
post-filter on recall by 23 pp at the cost of ~3× latency, because each
segment search is run with a generous per-node `ef`.

## State of the Art

Range filtering breaks the standard ANN graph contract. The graph indexes a
geometric neighborhood structure; the filter is *not* geometric.

| Approach | Idea | Failure mode |
|---|---|---|
| **Pre-filter brute** | Scan only items in range. | Useless for large/medium ranges — linear in subset size. |
| **Post-filter graph** | Search global graph, drop out-of-range hits. | Recall collapses when subset is small (paper: <10% selectivity). |
| **Milvus partition-key** | Pre-partition data on key; rewrite range as union of partitions. | Coarse — recall depends on partition granularity; rebuild on schema change. |
| **Filtered-DiskANN** | Walk graph but mask out-of-range edges during traversal. | Connectivity gaps for narrow ranges → poor recall (Wang et al., 2023). |
| **ACORN** (Patel 2024) | Predicate-aware graph with edge selection. | Built for categorical, not range; recent and complex. |
| **iRangeGraph** (Xu et al., SIGMOD 2024) | Segment tree of HNSWs over rank-space. | O(log n) graphs per item ⇒ memory blow-up vs single graph. |
| **SeRF** (Zuo & Deng, VLDB 2024) | Compress segment-tree graphs into one supergraph by tagging edges with the rank-interval where they are valid. | Compression bookkeeping is non-trivial; insertion is harder. |

The simplification we ship: a transparent segment tree of NSW graphs
(`iRangeGraph`-style), with the SeRF edge-tagging compression flagged as
future work (see roadmap below). This is exactly what the SeRF paper's
Section 5 evaluates as the iRangeGraph baseline, so the numbers are directly
comparable to published work.

## Design

### Data layout

```
items: { vector: Arc<Vec<Vec<f32>>>, key: Vec<f32> }
sorted = items by key (ascending)         # rank-space lives in [0, n)
segtree over rank-space, size = next_pow2(n), 2*size slots
node[i] owns Nsw over `sorted[span(i)]` if span(i) ≥ leaf_size
```

`Arc<Vec<Vec<f32>>>` makes the vector store shared across every NSW; we never
duplicate the embeddings. Each NSW stores its local→global id map and an
adjacency list. Empirical memory at N=20 000 / D=128 / `m=16` / `leaf_size=256`:

* embedding store: 20 000 × 128 × 4 B = 10.24 MB (the data itself)
* nsw-postfilter graph: 3.34 MB adjacency (1 graph)
* serf-segment-graph: 28.6 MB adjacency (160 graphs) → **8.6× graph overhead**

This matches the theoretical `log(n / leaf_size) ≈ log(20000/256) ≈ 6.3` and
the constant-factor overhead of small graphs near the leaves.

### Search

```
fn search(q, range, k):
    (rl, rr) = rank_range(range)               # 2× binary search on sorted keys
    if rr - rl ≤ leaf_size: brute-force the slice
    nodes   = canonical(rl, rr)                # ≤ 2·log₂(n) of them
    candidates = ∪ node.nsw.search(q, ef≥k)
    return top-k of dedup(candidates)
```

`canonical` returns the standard set of fully-covered segment-tree nodes for
`[rl, rr)`. Because each NSW already sees only its assigned items, the
returned candidates are *guaranteed in-range* — no postfiltering needed.

### Trait surface

```rust
pub trait RangeAnn {
    fn search(&self, q: &[f32], range: Range, k: usize) -> Vec<(usize, f32)>;
    fn name(&self) -> &'static str;
}
```

Three impls ship: `flat::Flat` (ground truth), `nsw_post::NswPost`,
`segment::SegmentGraph`. New backends (compressed SeRF, filtered DiskANN,
ACORN) plug in by implementing the same two methods, so benchmarks remain
apples-to-apples.

## Implementation notes

* **No external crates.** The crate has zero dependencies, including
  dev-deps. The NSW graph is ~150 lines of hand-rolled beam search with
  binary-heap candidates/results and bounded-degree pruning.
* **Distance:** squared L2. Inner product / cosine are one trait-extension
  away (left for the per-domain follow-up crate).
* **Determinism:** insertion order = id order, no RNG anywhere, so build is
  reproducible run-to-run.
* **`HashSet` is the GC bottleneck** in beam search; an open-addressing visit
  table keyed on local-id would shave another 20–30% off latency. Left for a
  follow-up so the first cut keeps to <500 LoC/file.

## Benchmark methodology

* **Workload:** 20 000 deterministic LCG-generated 128-d vectors in
  [-1, 1]^128, keys uniformly distributed in [0, 1).
* **Queries:** 200 fresh LCG vectors. For each selectivity `s`, the range is
  centred at `i/NQ` with half-width `s/2`, clamped to [0, 1]. This stresses
  the index across the full key domain rather than always hitting the middle.
* **Ground truth:** `flat::Flat` brute-force over the exact same range
  produces `Recall@k` for the other backends.
* **Hardware:** Apple Silicon (M-class), single thread, release build
  (`opt-level=3`, default codegen-units, no LTO bump). Numbers above are
  one run; variance across three runs was within ±8% on per-query latency,
  recall ±0.01.
* **Reproduction:** `cargo run --release -p ruvector-serf --example range_bench`

## Results — what they mean

1. **Postfilter recall collapse is real and severe.** At 1% selectivity
   NSW+postfilter recall is 0.085. Even with overscan = 8× the graph cannot
   reliably find 10 items that satisfy the predicate — it returns the global
   nearest, of which only ~0.8 (= 80 × 1%) are in range on average.
2. **Segment-graph trades memory for recall+latency.** 8.6× extra adjacency
   memory buys 1.00 recall at 1% selectivity *and* an 18× speedup vs
   postfilter, because each segment NSW only has ~200–2 000 items to search.
3. **Build cost is the headline cost.** Segment-graph build was 10.5 s vs
   2.1 s for the single graph — a 5× hit. For workloads where data is
   rebuilt nightly (logs/embeddings) this is fine; for high-churn streams
   it isn't, motivating the SeRF compression work in the roadmap.
4. **Brute is faster than you'd think at low selectivity.** At 1% selectivity
   brute over 200 items in cache-line order is 17 µs — only 2× slower than
   our segment-graph and still 9× faster than NSW+postfilter. **A
   production engine should fall back to brute below ~1k items** — the
   crate already does this via `leaf_size`.

## Practical failure modes

* **Skewed key distributions.** Rank-space partitioning is uniform in rank,
  not key. With heavy skew, a logical range like "last 24 h" might span a
  huge rank-range while a logical range "all-time" spans the whole set.
  Mitigation: rank-space is the right unit for cost-balancing — selectivity
  in rank is what determines work.
* **Updates.** Today every insert means rebuilding O(log n) graphs. SeRF's
  compressed form (one graph with edge interval tags) is the only known fix
  for in-place updates; we punt to the roadmap.
* **Filter that is not a range.** This crate is *only* for 1-D range
  filters. Multi-dimensional ranges → either a segment tree per dimension
  (works for 2-3 dims) or fall back to ACORN-style predicate-aware graphs.
* **`leaf_size` is workload-dependent.** Too small ⇒ memory blow-up + slow
  build; too big ⇒ brute-force fallback dominates narrow-range queries.
  A heuristic that grows `leaf_size` with `D` (cache-line economics) is
  worth a benchmark sweep — open follow-up.

## What to improve next (roadmap)

1. **SeRF edge-interval compression.** Replace the 160 small graphs with a
   single global graph whose edges carry rank-intervals where they are
   "active". Expected: 4–8× memory reduction; insertion still ugly.
2. **`u32` open-addressing visit table** in NSW search to drop the `HashSet`
   allocator dependency on every query. Expected: ~25% latency drop.
3. **Quantized embeddings.** With RaBitQ already in tree (`crates/ruvector-rabitq`),
   the segment-graph adjacency search can compute candidate distances in
   binary then refine top-`ef` in float. Expected: 3–5× per-query speedup.
4. **Parallel segment search.** Each canonical node is independent; rayon
   over canonical nodes gives a near-linear speedup up to `log n` threads.
5. **Multi-key range** via interval graph or KD-segment-tree.
6. **Streaming inserts** (the SeRF dynamic variant) so we can compete with
   pgvector + WAL workloads.

## Production crate layout proposal

When this graduates from nightly research:

```
crates/ruvector-serf/
  Cargo.toml                  # adds optional features: parallel, rabitq, mmap
  src/
    lib.rs                    # public API only — RangeAnn trait, Range, recall
    distance.rs               # L2 / IP / cosine behind a trait
    nsw/
      mod.rs                  # current Nsw
      visit.rs                # open-addressing visit table (replaces HashSet)
    segment.rs                # current SegmentGraph + edge-interval variant
    compress.rs               # SeRF-style edge-interval compression
    parallel.rs               # rayon-driven canonical-node search
    serde.rs                  # mmap-friendly serialization
  benches/                    # criterion benches that import the example workload
  examples/
    range_bench.rs            # the current realistic micro-bench
    timeseries_bench.rs       # workload generator that mimics log/embedding streams
```

## "How it works" walkthrough (blog version)

Imagine 20 000 product embeddings ordered by price. A customer asks: "what's
closest to my query, but only between $30 and $50?" 800 products satisfy that
predicate. A normal HNSW will happily return you the 10 closest products
*overall*, of which on average 0.4 will be in the price band — a recall of
4% if you only kept 10 results. Crank up `ef` and you waste compute walking
a graph that doesn't know anything about price.

SeRF flips the problem: sort the data by price once, then build a *tree of
graphs*. The root graph indexes everything; its left child indexes the
cheaper half; the right child the pricier half; and so on, recursively, until
each leaf graph holds a few hundred items. A query for "[$30, $50]" walks
the segment tree, picks the handful of subtree-graphs that exactly cover
that price range, and searches each. The candidates are *automatically*
in-range, because each graph only ever saw items from its own slice of price.

The price of this trick is memory: each item sits in `log(n / leaf_size)`
graphs instead of one. For 20 000 items at leaf size 256 that's about 6
copies of each adjacency list — ~9× the memory of a single HNSW. In return
you get a graph that is correct by construction over every possible range,
and is *faster* for narrow ranges because each segment graph is small.

## References

1. C. Zuo, F. Deng. *SeRF: Segment Graph for Range-Filtering Approximate
   Nearest Neighbor Search.* VLDB 2024.
2. C. Xu, M. Patel, R. Wang et al. *iRangeGraph: Improvising Range-dedicated
   Graphs for Range-Filtering Nearest Neighbor Search.* SIGMOD 2024.
3. P. Patel et al. *ACORN: Performant and Predicate-Agnostic Search Over
   Vector Embeddings and Structured Data.* SIGMOD 2024.
4. R. Wang et al. *Filtered-DiskANN: Graph Algorithms for Approximate Nearest
   Neighbor Search with Filters.* WWW 2023.
5. Y. Malkov, D. Yashunin. *Efficient and robust approximate nearest neighbor
   search using Hierarchical Navigable Small World graphs.* TPAMI 2020.
