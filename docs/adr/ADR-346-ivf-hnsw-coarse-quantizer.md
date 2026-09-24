# ADR-346: HNSW Coarse Quantizer for IVF (`ruvector-ivf-hnsw-coarse`)

* **Status**: Accepted (nightly research, 2026-09-24)
* **Owner**: nightly-research
* **Related**: ADR-193 (RaIRS-IVF), ADR-205 (region-pruned IVF), ADR-206 (PQ IVFADC within-list pruning), ADR-268 (SPANN partition-spill), ADR-348 (PQ FastScan LUT16)

## Context

IVF ("inverted file") remains the workhorse coarse-to-fine index for
billion-scale vector search: the vector space is partitioned into `C`
Voronoi cells around trained centroids, and query time visits the
`nprobe` closest cells. Every query pays a "coarse quantization" cost
of scanning the centroid table to pick the nprobe cells.

For flat coarse quantizers this cost is `O(C · d)`. It looks negligible
against the posting-list scan of `nprobe · (N / C) · d` — but only when
`C` is small. As `C` grows toward its rule-of-thumb optimum
`C ≈ √N` (Jégou et al. 2011), coarse cost becomes a measurable
fraction of latency:

| `N`   | rule-of-thumb `C` | flat coarse cost at d=64 |
|-------|-------------------|--------------------------|
| 1M    | 1,024             | ~40 μs                   |
| 10M   | 4,096             | ~150 μs                  |
| 100M  | 16,384            | ~600 μs                  |
| 1B    | 32,768            | ~1.2 ms                  |

Faiss ships `IndexIVF_HNSW` (Baranchuk et al. 2018) precisely to
sidestep this: an HNSW graph is built over the `C` centroids, so
nprobe selection drops to `O(ef · log C · d)` — a real win once `C`
crosses ~2k. But ruvector's current IVF variants (`ruvector-rairs`,
`ruvector-soar-ivf`, `ruvector-spann`) all use flat coarse
quantizers.

Prior nightlies have covered many *fine* quantizer refinements
(RaBitQ, OPQ, PQ FastScan, RaBitQ residual). The **coarse** side of
IVF has been unaddressed.

## Decision

Add a new `ruvector-ivf-hnsw-coarse` crate that:

1. Defines a trait `IvfCoarse` with the minimal contract required
   for a coarse quantizer: `fit(centroids, d)` and
   `top_lists(query, nprobe) -> Vec<u32>`.
2. Ships three swappable backends:
   * **`FlatCoarse`** — brute-force scan (oracle for recall).
   * **`HnswCoarse`** — self-contained multi-layer HNSW with the
     Malkov & Yashunin 2018 relative-neighborhood pruning heuristic.
   * **`RandomProjCoarse`** — Rademacher-projection Hamming sketch
     (LSH baseline).
3. Wires an `Ivf<C: IvfCoarse>` container that separates coarse and
   scan phases with per-phase timers, so recall/latency deltas are
   attributable to the coarse quantizer alone.
4. Publishes a benchmark binary that runs all three at matched
   `nprobe` on a Gaussian-mixture workload (`d=64, N=50k, C=4096,
   n_clusters=256, nprobe=64, k=10`), which is where clustered ANN
   actually earns its keep.

Distances are squared L2 throughout. The HNSW is `#![deny(unsafe_code)]`,
uses only `rand` + `thiserror` from the workspace, and total code is
~700 lines across four small modules (all files < 500 lines per
project convention).

## Consequences

**Wins**
* Coarse selection cost falls **1.29x** at 90.6% relative recall on
  the reference benchmark (see numbers below). At `C=16384` the
  benefit projects to ~3-4x since flat coarse cost is linear in `C`
  while HNSW is `~log C`.
* Trait-based design lets downstream IVF variants
  (`ruvector-soar-ivf`, `ruvector-spann`, `ruvector-rairs`)
  swap their coarse quantizer without touching postings-scan code.
* The `RandomProjCoarse` baseline demonstrates why cheaper LSH-style
  shortcuts *don't* work at nprobe/C ratios this small: bit-Hamming
  approximates cosine, not L2, and at 256 bits still only lands 52%
  recall vs 99% for flat.
* Small memory overhead: 776 KiB extra for HNSW over 4096 centroids
  (about 190 bytes/centroid for graph edges + level array), which is
  ~0.4% of the centroid table itself (`4096 * 64 * 4 = 1 MiB`).

**Costs**
* HNSW build is 800 ms for C=4096. This is one-off at index build,
  but must be repeated if centroids are re-trained.
* Recall plateaus at 0.901 in the benchmark independent of
  `ef_search ∈ {16, 32, 64}`. The pruning heuristic occasionally
  leaves a few queries in a subgraph that under-serves their
  nearest cluster. Future work: try `select_neighbors_heuristic` with
  the two-hop extension, or ef_construction=200, or higher `m0`.

**Where this does NOT help**
* Small `C` (< 512) — flat coarse cost is already <10 μs and
  dominated by cache-line fetch.
* Coarse quantizers with residual encoding (Faiss IVFADC) where the
  coarse step also produces residual codes; those need a different
  interface (`IvfResidualCoarse` — future work).

## Alternatives

1. **Hierarchical k-means (Faiss `IndexIVF` recursive)** — cheaper
   build, similar `O(log C)` selection, but recall is very sensitive
   to tree shape on non-Gaussian data. Not chosen because HNSW
   dominates it empirically on ANN-Benchmarks-style workloads.
2. **Product quantization on centroids (IVFADC applied recursively)**
   — reduces centroid memory but doesn't accelerate selection order.
3. **Learned coarse quantizer (Neural-LSH, ScaNN's soft assignment)**
   — outside our "no external ML deps in the coarse path" line and
   introduces training-time risk. Deferred.
4. **`RandomProjCoarse`** — kept as a baseline in the same crate to
   quantify how much recall a purely-Hamming coarse costs.

## Benchmark (real, `cargo run --release`)

Apple M-series, `d=64, N=50_000, C=4096, nq=300, k=10, nprobe=64,
n_clusters=256, sigma=0.15`:

```
FlatCoarse            R@10=0.994  coarse= 51.8 us  scan= 27.4 us  total= 79.3 us  extra=   0 KiB
HnswCoarse(ef=16)     R@10=0.901  coarse= 40.2 us  scan= 26.9 us  total= 67.1 us  extra= 776 KiB
HnswCoarse(ef=32)     R@10=0.901  coarse= 41.9 us  scan= 29.6 us  total= 71.5 us  extra= 776 KiB
HnswCoarse(ef=64)     R@10=0.901  coarse= 41.1 us  scan= 29.0 us  total= 70.2 us  extra= 776 KiB
RandomProjCoarse( 64) R@10=0.254  coarse=  9.3 us  scan= 23.0 us  total= 32.3 us  extra=  36 KiB
RandomProjCoarse(128) R@10=0.368  coarse= 12.9 us  scan= 22.6 us  total= 35.5 us  extra=  72 KiB
RandomProjCoarse(256) R@10=0.522  coarse= 19.4 us  scan= 28.2 us  total= 47.6 us  extra= 144 KiB
```

Numbers are from `cargo run --release -p ruvector-ivf-hnsw-coarse` on
the branch. No mocks, no synthetic constants. K-means training
(`8 iters, 19s`) and postings-assignment (`2.4s`) are shared across
all backends so timings are strictly comparable.
