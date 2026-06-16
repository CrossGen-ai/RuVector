# ruvector-segment-rangehnsw

Range-filtered approximate nearest neighbor search using a segment tree of
small proximity graphs — an iRangeGraph-style (SIGMOD 2024) PoC for the
`ruvector` workspace.

## Why

Classical HNSW + post-filter loses recall catastrophically when the range
filter is selective: at 1% selectivity our flat HNSW + post-filter measured
**6.5% recall@10** because almost no in-range neighbors survived the beam.
Pre-filter (predicate-agnostic expansion, ACORN-style) keeps recall high but
becomes 100× slower than brute force on narrow ranges because it walks the
graph globally.

`segment-rangehnsw` pre-indexes O(log n) overlapping windows over the range
key. A query [lo, hi] descends the tree, finds O(log n) fully-covering
windows, and runs a small graph search inside each. Distance work scales
roughly with the actual selectivity, so recall and QPS both stay high across
the selectivity sweep.

## Numbers (release build, M2 Pro, n=10 000, D=64, k=10, 100 queries)

| Selectivity | Variant                  | QPS     | Recall@10 |
|-------------|--------------------------|---------|-----------|
| 1%          | flat HNSW + post-filter  |  22 502 | 0.065     |
| 1%          | flat HNSW + pre-filter   |     841 | 0.995     |
| 1%          | **segment-rangehnsw**    | 230 858 | **1.000** |
| 5%          | flat HNSW + post-filter  |  20 067 | 0.373     |
| 5%          | flat HNSW + pre-filter   |   1 994 | 0.987     |
| 5%          | **segment-rangehnsw**    |  27 859 | **1.000** |
| 20%         | flat HNSW + post-filter  |  20 609 | 0.733     |
| 20%         | flat HNSW + pre-filter   |   4 353 | 0.946     |
| 20%         | **segment-rangehnsw**    |  10 264 | **0.985** |
| 50%         | flat HNSW + post-filter  |  17 647 | 0.786     |
| 50%         | flat HNSW + pre-filter   |   7 447 | 0.876     |
| 50%         | **segment-rangehnsw**    |   6 789 | **0.969** |

Memory: 3.79 MiB flat graph vs 29.92 MiB segment tree (7.9× — O(log n) factor
for redundant windows).

## Run

```
cargo build --release -p ruvector-segment-rangehnsw
cargo test  --release -p ruvector-segment-rangehnsw
cargo run   --release -p ruvector-segment-rangehnsw --example bench
```

## Layout

- `src/dist.rs` — distance, deterministic LCG, Gaussian sampling
- `src/graph.rs` — single-layer proximity graph with beam search +
  pre-filter expansion
- `src/segment.rs` — segment-tree of graphs
- `src/lib.rs` — public API, brute-force ground truth, integration tests
- `examples/bench.rs` — four-variant comparison harness

See `docs/research/nightly/2026-06-16-segment-hnsw-range-filter/README.md` for
the SOTA survey, design notes, and the full benchmark walk-through.
