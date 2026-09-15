# ruvector-pdx-scan

PDX (dimension-partitioned) vector-scan layout for ruvector, following
Kuffo, Krommydas, Boncz — *"PDX: A Data Layout for Vector Similarity
Search"*, SIGMOD 2024.

## Idea

Instead of storing each vector's `d` dimensions contiguously (row-major /
"horizontal"), group vectors into blocks of 64 and store, within a block,
dimension `j`'s value for **all 64 vectors contiguously**. A block is a
`d × 64` column-major tile. Scanning becomes: load one column (64 lanes,
1–2 SIMD registers), do a broadcast subtract-and-FMA against the query
scalar for that dim, accumulate into 64 per-lane running sums. No shuffles,
no cross-lane reductions in the hot loop.

Optionally, after every `dim_step` dimensions, compare each lane's partial
distance to the current top-k threshold to skip doomed lanes. Choose the
dimension order to make the threshold tighten fast (variance-descending is
a good default).

## What's in this crate

* `HorizontalStore` / `PdxStore` — two backends behind the same `VectorStore` trait.
* `DimensionOrder::{Identity, VarianceDesc}` — permutation policies.
* Three brute-force scans:
  * `brute_horizontal` — baseline row-major.
  * `brute_pdx` — PDX layout, full traversal, no pruning.
  * `brute_pdx_pruned` — PDX + per-lane threshold pruning.
  * `brute_pdx_block_pruned` — PDX + branchless block-level early exit.
* `IvfHorizontal` / `IvfPdxPruned` — IVF drivers that stack a k-means
  centroid index on top of either backend.
* `pdx-bench` binary — real end-to-end numbers on random Gaussian data
  (n=50 000, d=128, k=10, 100 queries).

## Measured numbers (Apple M-series, `--release`, single thread)

```
== PDX brute-force ==
n=50000, d=128, k=10, queries=100
  horizontal          :  1.624 ms/query   1.00x   (baseline)
  pdx plain           :  0.553 ms/query   2.94x   recall 1.000
  pdx pruned (lane)   :  4.722 ms/query   0.34x   recall 1.000   work 0.718
  pdx block-prune     :  0.874 ms/query   1.86x   recall 1.000   work 0.959

== IVF nprobe (64 lists) ==
nprobe=16
  ivf horizontal      :  0.465 ms/query   recall 0.513
  ivf pdx-pruned      :  1.261 ms/query   recall 0.513
```

**Headline:** the PDX layout alone yields a 2.94× brute-force speedup with
zero recall loss. Per-lane pruning is a **net loss** on this workload
because the `if active[lane]` predicate defeats autovectorization; the
branchless block-level early-exit variant recovers most of the layout win
(1.86×) while still skipping the last dims of doomed blocks. See the
research doc for the full picture.

## Run

```bash
cargo test  --release
cargo run   --release --bin pdx-bench
```

## License

MIT.
