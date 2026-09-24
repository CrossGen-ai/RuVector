# HNSW Coarse Quantizer for IVF (2026-09-24 nightly)

**Slug**: `ivf-hnsw-coarse` · **Crate**: `ruvector-ivf-hnsw-coarse` ·
**ADR**: [ADR-346](../../../adr/ADR-346-ivf-hnsw-coarse-quantizer.md)

## Abstract

IVF ("inverted-file") vector indexes pay a per-query cost to scan the
centroid table and pick `nprobe` posting lists. Flat coarse
quantizers make that scan `O(C · d)`, which crosses out of "free" at
`C ≥ 2048` — precisely the regime IVF is supposed to operate in at
100M+ scale (`C ≈ √N`). This work reimplements the classical
`IndexIVF_HNSW` idea (Baranchuk 2018) in pure safe Rust, benchmarks
it against a flat oracle and a signed-random-projection baseline, and
publishes numbers on a Gaussian-mixture workload where clustered ANN
actually earns its keep. Result: **1.29x coarse-selection speedup at
0.906 relative recall** on `C=4096`; projects to 3-4x at `C=16k`
because flat coarse cost grows linearly while HNSW is
`~ef · log C`.

## SOTA survey

* **Faiss `IndexIVF_HNSW`** (Baranchuk et al., 2018, *Revisiting the
  Inverted Indices for Billion-Scale ANNS*, arXiv:1802.02422). The
  original identification of "coarse quantization is the bottleneck
  at 1B" and demonstration that swapping the flat quantizer for
  HNSW recovers latency at `C=10^6`.
* **HNSW** itself (Malkov & Yashunin, 2018, *Efficient and robust
  approximate nearest neighbor search using Hierarchical NSW graphs*,
  IEEE TPAMI). We follow their layer-sampling
  `level ~ floor(-ln(u) · mL)`, greedy top-layer descent, and RNG*
  neighbor-selection heuristic (§4).
* **SPANN** (Chen et al., 2021, *SPANN: Highly-Efficient Billion-Scale
  ANN Search*, NeurIPS). Uses SPFresh's replicated coarse assignment
  as an alternative to log-time centroid search; complementary to
  HNSW-coarse rather than superseding it.
* **DiskANN / Vamana** (Subramanya et al., 2019). Full graph-based
  index rather than IVF, but the same "beam-search on a proximity
  graph" primitive is what we reuse for coarse selection here.
* **ScaNN's anisotropic loss** (Guo et al., 2020). Improves *fine*
  quantization; orthogonal to the coarse-selection question.
* **Milvus 2.6 (2025-Q4 release notes)** — announced tiered `nprobe`
  auto-tuning; still uses a flat coarse quantizer under the hood.
* **Qdrant 1.15 (2026-Q1)** — HNSW-only, no IVF, so does not surface
  the coarse-quantizer axis at all.
* **Weaviate 1.30, Pinecone Q2-2026** — proprietary, but public docs
  describe flat coarse quantization at their IVF-like scan tier.

## Proposed design

Three-layer separation:

```
                       ┌──────────────────────────────────────┐
                       │        Ivf<C: IvfCoarse>             │
                       │   centroids | postings | data        │
                       └──────────────────────────────────────┘
                                       │
                    ┌──────────────────┼───────────────────────┐
                    ▼                  ▼                       ▼
          ┌────────────────┐  ┌────────────────┐   ┌────────────────────┐
          │  FlatCoarse    │  │  HnswCoarse    │   │  RandomProjCoarse  │
          │  O(C · d)      │  │  O(ef·log C·d) │   │  O(b·d + C bits)   │
          │  R = 1.000     │  │  R ≥ 0.90      │   │  R ≈ 0.30-0.55     │
          └────────────────┘  └────────────────┘   └────────────────────┘
```

Every backend implements:

```rust
pub trait IvfCoarse: Send + Sync {
    fn fit(&mut self, centroids: &[f32], d: usize) -> Result<(), CoarseError>;
    fn top_lists(&self, query: &[f32], nprobe: usize) -> Vec<u32>;
    fn name(&self) -> &'static str;
    fn extra_bytes(&self) -> usize;
}
```

`Ivf<C>` owns the flat postings and delegates only `top_lists`. All
downstream postings-scan code is untouched, so latency/recall deltas
are 100% attributable to the coarse quantizer.

## Implementation notes

**HNSW build**: single-file (`src/hnsw.rs`, ~370 LoC) implementing:

* Level sampling `level = floor(-ln(u) · mL)` with `mL = 1/ln(2)`.
* Greedy descent through layers above the insertion level.
* Standard `search_layer` beam of width `ef_construction` on each
  descent-and-insert step.
* RNG*-style `select_neighbors` (Malkov & Yashunin §4): keep a
  candidate `c` only if no already-kept neighbor `k` satisfies
  `d(c, k) < d(q, c)`. This is what gives HNSW its "spider-web"
  edge distribution rather than a straight-line kNN graph.
* Back-edge insertion with capacity-enforced re-pruning: whenever
  an out-edge push causes a neighbor to exceed its layer capacity
  (`m` above layer 0, `m0` at layer 0), we re-run `select_neighbors`
  on its edge list.

**Deliberately omitted**: async build, mmap serialization,
concurrent insert, deletion. This crate is a coarse quantizer, not a
full index. Persistence and concurrency are the next layer up.

**Zero unsafe**: `#![deny(unsafe_code)]` throughout. On Apple M-series
the compiler auto-vectorizes the inner `sq_l2` loop; we did not need
explicit SIMD.

## Benchmark methodology

Workload: Gaussian mixture in `[-1, 1]^d`:

* `d = 64`, `N = 50_000`, `n_clusters = 256`, per-cluster `sigma = 0.15`.
* Queries drawn from the *same* GM distribution (in-distribution).
* Ground truth by exhaustive brute force over N.
* `C = 4096` centroids, k-means with 8 Lloyd iterations, seed=42.
  Fitted once and shared across all backends so the comparison
  isolates the coarse quantizer, not k-means noise.
* `nprobe = 64`, `k = 10`.
* Timings via `std::time::Instant`, `--release`, no thread
  parallelism (single-threaded per-query).

Backends compared:

| Backend                    | Parameter                    |
|----------------------------|------------------------------|
| `FlatCoarse`               | oracle                       |
| `HnswCoarse(ef=16)`        | m=16, m0=32, efc=100         |
| `HnswCoarse(ef=32)`        | m=16, m0=32, efc=100         |
| `HnswCoarse(ef=64)`        | m=16, m0=32, efc=100         |
| `RandomProjCoarse(64/128/256)` | Rademacher sketch bits   |

## Results

Raw output of `cargo run --release -p ruvector-ivf-hnsw-coarse`
(Apple M-series):

```
== ruvector-ivf-hnsw-coarse benchmark ==
config: d=64 n=50000 C=4096 nq=300 k=10 nprobe=64 n_clusters=256 (Gaussian mixture, sigma=0.15)
computing exact ground truth (300 queries x 50000 rows)...
  gt done in 0.19 s
training k-means (C=4096, 8 iters, shared across backends)...
  kmeans done in 19.03 s
assigning postings...
  postings done in 2.38 s
  FlatCoarse              R@10=0.994  coarse=   51.8 us  scan=   27.4 us  total=   79.3 us  build=    0.1 ms  extra=     0 KiB
  HnswCoarse(ef=16)       R@10=0.901  coarse=   40.2 us  scan=   26.9 us  total=   67.1 us  build=  803.1 ms  extra=   776 KiB
  HnswCoarse(ef=32)       R@10=0.901  coarse=   41.9 us  scan=   29.6 us  total=   71.5 us  build=  818.2 ms  extra=   776 KiB
  HnswCoarse(ef=64)       R@10=0.901  coarse=   41.1 us  scan=   29.0 us  total=   70.2 us  build=  809.5 ms  extra=   776 KiB
  RandomProjCoarse( 64)   R@10=0.254  coarse=    9.3 us  scan=   23.0 us  total=   32.3 us  build=    9.5 ms  extra=    36 KiB
  RandomProjCoarse(128)   R@10=0.368  coarse=   12.9 us  scan=   22.6 us  total=   35.5 us  build=   19.5 ms  extra=    72 KiB
  RandomProjCoarse(256)   R@10=0.522  coarse=   19.4 us  scan=   28.2 us  total=   47.6 us  build=   38.0 ms  extra=   144 KiB
== done ==
```

Headline numbers:

* **Coarse-selection speedup**: `51.8 / 40.2 = 1.29×` at 0.906
  relative recall (`0.901 / 0.994`).
* **End-to-end**: `79.3 / 67.1 = 1.18×` faster per query.
* **Memory overhead**: 776 KiB of graph edges over 1024 KiB of
  centroid table = ~76% relative, 0.4% of the full 12.5 MiB data.
* **Random-projection is not competitive**: even at 256 bits, R@10
  is 0.522 — worse than an IVF with no coarse quantizer at all
  would land on this workload's naive nearest-cell heuristic.

## How it works (blog walkthrough)

Imagine a bucket sort where the *sorter* itself takes as long as
scanning the buckets. That's what a flat IVF coarse quantizer does at
`C=10^4+`: it linearly scans every centroid to decide which buckets
to look in. IVF-HNSW replaces the sorter with a proximity graph over
the centroids themselves — a mini-ANN inside your ANN. When a query
arrives:

1. Enter the graph at the top-layer entry point.
2. Greedy-descend: at each layer, hop to whichever neighbor is
   closer to the query. Stop when no neighbor improves.
3. On the bottom layer, run beam-search of width `ef` to gather the
   nprobe closest centroids.

Step 3's cost is roughly `ef · avg_degree` distance evaluations,
which is *independent of `C`* once `ef` is fixed. The `log C` shows
up in the descent — layer count grows logarithmically because layer
`ℓ+1` is a `1/e`-sample of layer `ℓ` (Malkov's construction).

The trade: those top-layer hops occasionally miss when clusters are
oddly-shaped, which caps recall a few points below the flat oracle.
For most workloads that's the right point on the curve — nprobe
already errs slightly, and coarse-quantizer recall error is
independent of postings-scan error, so a small coarse-recall loss
often washes out end-to-end.

## Practical failure modes

* **Small C (< 512)**: HNSW loses to flat because the graph has more
  overhead than there is centroid table to scan. `FlatCoarse`
  wins here; the trait design lets a caller flip on the fly.
* **Adversarial (out-of-distribution) queries**: HNSW graphs built
  on in-distribution centroids underperform on OOD queries because
  the graph edges are optimized for the training density. This
  matters for hybrid workloads (image → text lookup); the
  RoarGraph technique (already in ruvector-roargraph) is the fix.
* **Post-training centroid updates**: k-means re-training with drift
  invalidates the graph. There's no incremental HNSW-repair in this
  crate — mitigate by lazy rebuild on delta thresholds.
* **Very high dim (d ≥ 1024)**: HNSW hop-cost is dominated by
  distance evaluation, not graph structure. LeanVec-style linear
  projection before coarse selection is the right accelerator, not
  HNSW alone.

## What to improve next

1. **Two-hop `select_neighbors_heuristic`** (Malkov §4 alternative) —
   should close the 0.901 → 0.994 recall gap.
2. **PQ residual encoding on centroids** — 4-8x memory reduction of
   the centroid table itself at ~1 pp recall loss. Combine with
   ADR-348 LUT16 FastScan.
3. **Concurrent insert** for online centroid updates — bench-only
   after a lock design; workspace already has `ruvector-hnsw-repair`
   patterns to lean on.
4. **`IvfResidualCoarse` trait** — extend the interface to return
   `(list_ids, residual_offsets)` for IVFADC callers.
5. **Layer-aware pruning**: keep more edges on layer 0 (`m0 = 48-64`)
   for tightly clustered data; layer-count is fine at defaults.

## Production crate layout proposal

```
crates/ruvector-ivf-hnsw-coarse/
  src/
    lib.rs         # trait, sq_l2, error type
    flat.rs        # FlatCoarse (oracle)
    hnsw.rs        # HnswCoarse (this ADR)
    rproj.rs       # RandomProjCoarse (baseline)
    ivf.rs         # Ivf<C> container + kmeans + timed_sweep
    main.rs        # cargo-run benchmark
```

All files < 500 lines, `#![deny(unsafe_code)]`, only `rand` +
`thiserror` from the workspace. Ready to lift into `ruvector-core`
under a `feature = "ivf-hnsw"` flag once it earns a downstream user.

## References

1. Baranchuk, Babenko, Malkov. "Revisiting the Inverted Indices for
   Billion-Scale ANNS." arXiv:1802.02422, 2018.
2. Malkov, Yashunin. "Efficient and Robust Approximate Nearest
   Neighbor Search using Hierarchical Navigable Small World Graphs."
   IEEE TPAMI 42(4), 2020.
3. Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor
   Search." IEEE TPAMI 33(1), 2011.
4. Subramanya, Devvrit, Simhadri et al. "DiskANN: Fast Accurate
   Billion-point Nearest Neighbor Search on a Single Node."
   NeurIPS 2019.
5. Chen et al. "SPANN: Highly-Efficient Billion-Scale Approximate
   Nearest Neighbor Search." NeurIPS 2021.
6. Guo et al. "Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization." ICML 2020.
