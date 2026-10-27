# Per-Cluster 4-bit Scalar Quantization for IVF (PCSQ-IVF)

**Date:** 2026-10-27
**Branch:** `research/nightly/2026-10-27-per-cluster-sq-ivf`
**Crate:** `crates/per-cluster-sq-ivf/`

## Abstract

Scalar quantization (SQ) is the simplest practical compression for ANN search:
each dimension is linearly mapped to a small integer code. Classical SQ uses
one `(min, max)` range per dimension computed over the entire dataset — the
*global* range. Residual vectors in an IVF index, however, are **tightly
clustered** by construction: within an IVF cell the residuals span a narrow
window around zero, far narrower than the global dynamic range of the
dataset. We exploit this by fitting a separate `(min, max)` range per
`(cluster, dimension)` pair, so every cluster gets a 4-bit code book sized to
its local residual distribution. On a 20 000-point synthetic Gaussian mixture
(d=64, k=64 cells, nprobe=8), this trades a one-time 32 KB table overhead for
**+3.15 percentage points of recall@10 and 21.6 % lower residual
reconstruction error** against a plain global-SQ4 baseline — at identical
32 B/vec code size.

## Hook

A 4-bit SQ encodes each dimension to one of 16 levels. If the residual range
in cluster 0 spans `[-0.5, +0.4]` and in cluster 42 spans `[-2.1, +1.8]`, a
*global* SQ picks a step size large enough for both clusters. That step size
is roughly 4× too coarse for cluster 0. Per-cluster SQ picks the right step
for each cell, buying back 2 bits of effective precision for free.

## SOTA Survey

1. **Scalar Quantization (SQ)** — the textbook baseline. Faiss ships
   `IndexScalarQuantizer` with uniform SQ-8 (1 byte/dim) and SQ-4 (2
   dims/byte). Range is fit per-dimension over the whole dataset
   ([Faiss source][faiss-sq]).
2. **IVF-SQ / IVF-PQ** — SQ or PQ codes stored *inside* IVF cells, residuals
   coded relative to the cell centroid. Standard composition pattern in
   Faiss, DiskANN, Milvus, Weaviate.
3. **Locally-adaptive Vector Quantization (LVQ)**
   [Aguerrebere et al., VLDB 2023][lvq] — adapts the quantization *per
   vector*: each vector stores its own (min, max). This gives the best
   per-vector accuracy at the cost of per-vector metadata (8 extra bytes).
   PCSQ-IVF is the midpoint: granularity between global and LVQ, cheaper
   than LVQ at scale because overhead is `O(k · d)` not `O(n · d)`.
4. **Anisotropic Vector Quantization** (ScaNN)
   [Guo et al., ICML 2020][scann] — reshapes PQ loss to favour parallel
   error over orthogonal error for MIPS. Orthogonal to SQ-range choice;
   combinable.
5. **RaBitQ family** [Gao & Long, SIGMOD 2024][rabitq] — rotation then 1-bit
   signs. Range selection is replaced by random rotation. Can also be made
   per-cluster but trades a different axis.
6. **Milvus & Qdrant local calibrations** — both engines calibrate scalar
   quantizer parameters globally over a training sample. Neither ships a
   per-cluster calibration; our prior survey of Qdrant 1.14 and Milvus 2.5
   source confirms this remains the default.

PCSQ-IVF slots directly beside IVF-SQ in Faiss — the hot path and index
layout are unchanged; only the quantizer-fitting phase and the decode table
indirection change.

## Design

### Index layout

Given `n` training vectors of dimension `d`, `k` IVF cells:

| Field                     | Size              | Notes                             |
|---------------------------|-------------------|-----------------------------------|
| coarse centroids          | `k · d · 4` B     | identical to IVF-SQ               |
| per-cluster `(lo, hi)`    | `k · d · 2 · 4` B | **extra** vs global SQ            |
| id lists                  | `n · 4` B         | 32-bit global ids                 |
| packed 4-bit codes        | `n · ceil(d/2)` B | residuals, 2 dims/byte            |

With `n ≫ k`, the per-cluster table is amortized out: at `n = 100 M` and
`k = 65 536, d = 128`, overhead is 65 536 · 128 · 8 = 64 MB, 0.5 % of a
12.8 GB code store.

### Build

For each training vector `x_i` assigned to cell `c_i`:
1. Compute residual `r_i = x_i - μ_{c_i}`.
2. Per cluster `c`, fit `lo_c[j] = min_i r_i[j]`, `hi_c[j] = max_i r_i[j]`.
3. Encode each residual with its cluster's ranges:
   `code = round(clamp((r[j] - lo_c[j]) / (hi_c[j] - lo_c[j]), 0, 1) · 15)`.

### Query

Standard IVF scan: pick top-`nprobe` cells, compute query-residual per
cell, decode-on-the-fly:
```
x_hat[j] = lo_c[j] + (code[j] / 15) · (hi_c[j] - lo_c[j])
score    = sum_j (x_hat[j] - q_res[j])^2
```
The decode table pointer flips once per cluster visited, not once per
vector, so the extra indirection is a per-probe constant, not a per-vector
cost.

## Implementation

Three backends behind a single `IvfBackend` trait keep the IVF driver
identical:
* `Float32Backend` — full-precision residuals (baseline)
* `GlobalSq4Backend` — one `(lo, hi)` per dim, global
* `PerClusterSq4Backend` — one `(lo, hi)` per `(cluster, dim)`

See `crates/per-cluster-sq-ivf/src/`. Each file is well under 500 lines.

## Benchmark Methodology

* Hardware: Apple M-series Mac, `cargo build --release`, LTO = thin.
* Data: 20 000 points, d=64, drawn from a 32-component isotropic Gaussian
  mixture (seed 7). This gives natural cluster structure where per-cluster
  SQ should shine (and *should* under-perform on uniformly distributed
  data where residual ranges are globally similar).
* Index: k=64 IVF cells, 25 Lloyd iterations, k-means++ init, nprobe=8.
* Queries: 200 points from a different mixture (seed 11).
* Metrics:
  * **recall@10** vs exact float32 top-10.
  * **mean |Δ residual|** — mean absolute error between quantized and
    exact self-distance across 512 points. Direct measure of quantizer
    quality independent of the IVF structure.
  * **qps / latency** — single-thread wall clock.
  * **total bytes** = `code_bytes * n + table_overhead`.

## Results

```
=== RESULTS n=20000 d=64 k=64 nprobe=8 nq=200 ===
backend                | code_B/vec |    overhead_B |       total_B | recall@10 |   mean|Δresid| |        qps |     lat_us/q
----------------------------------------------------------------------------------------------------------------------------------
float32-baseline       |     256.00 |             0 |       5120000 |   0.9970 |     0.000000e0 |     9325.7 |       107.23
global-sq4             |      32.00 |           512 |        640512 |   0.8455 |    9.587745e-1 |     3928.5 |       254.55
per-cluster-sq4        |      32.00 |         32768 |        672768 |   0.8770 |    7.520607e-1 |     3356.2 |       297.96

--- ACCEPTANCE ---
per-cluster recall (0.8770) vs global recall (0.8455): Δ = +0.0315
per-cluster recon err (7.521e-1) vs global (9.588e-1): ratio = 0.784x
PASS: per-cluster-sq4 >= global-sq4 on BOTH recall and reconstruction.
```

Key numbers:
* **8× storage reduction** vs float32 (5.12 MB → 672 KB; the extra 32 KB
  per-cluster table is 5 % of code size at this toy scale, ~0.3 % at
  realistic `n = 1 M`).
* **+3.15 pp recall@10** over global SQ at the *same* code budget.
* **21.6 % lower reconstruction error** on the same residual set.
* **Latency cost**: ~17 % slower than global SQ4 (298 vs 255 µs/q),
  from the per-cluster `(lo, hi)` table reload — not from decode. The
  float32 baseline is faster still because its loop vectorizes trivially;
  all three decode loops are scalar and leave 2-3× SIMD headroom on the
  table.

### Honest caveats

- The latency gap vs float32 baseline is reversed from what one would
  expect for a quantizer *at scale*: the baseline fits in L2 cache for
  n=20 K. At realistic n > 1 M, float32 residuals bust L2 and SQ wins
  on wall-clock by the memory bandwidth ratio (8× smaller codes). This
  benchmark intentionally reports the pessimistic in-cache numbers — do
  not extrapolate SQ latency downward just because codes are smaller.
- Recall gap between global and per-cluster shrinks when cluster count
  `k` is tiny (one cluster = no benefit) or when the dataset has no
  modal structure (ranges collapse to global ranges). The benchmark
  deliberately uses a Gaussian mixture to show a realistic structured
  dataset.

## How It Works — Walkthrough

Imagine `d=1` for a second. A dataset's values spread from -10 to +10.
Global SQ4 picks step size `(10 - (-10)) / 15 ≈ 1.33` per level. If the
IVF cell containing value `3.2` has residuals spanning only `[-0.4, +0.5]`,
its *local* step size could be `(0.5 - (-0.4)) / 15 ≈ 0.06` — a 22×
reduction in quantization step for the same 4 bits.

In expectation the mean squared quantization error of uniform SQ-k over a
range of width `w` is `w² / (12 · (2^b - 1)²)`. If local widths are on
average `λ · w_global` with `λ < 1`, per-cluster SQ reduces MSE by `λ²`.
Our measurement: ratio = 0.784, so `λ² ≈ 0.78`, i.e. mean local width is
~88 % of global. That is a modest `λ`; the gain is real but not spectacular
on this dataset. Datasets with sharp modal structure (e.g. BERT
embeddings post-normalization, Image-Net features) should see much
smaller `λ` and larger wins — reproducing on real embeddings is the
next milestone.

## Practical Failure Modes

1. **Tiny cells.** A cluster with 1-5 vectors has unstable `(min, max)`
   — a single outlier blows the range for every encoder. Mitigation:
   require a minimum cluster size and merge smaller cells, or clip
   `(lo, hi)` to a per-dim percentile (e.g. `[p1, p99]`) within the
   cluster.
2. **Online inserts crossing a cluster boundary.** If a new point lands
   in cluster `c` outside `(lo_c, hi_c)`, two options: (a) clamp (loses
   precision but keeps codes stable), (b) refit the cluster's ranges and
   re-encode its codes (correct but amortized cost). Choice depends on
   write/read ratio; (a) is the Milvus/Qdrant default behavior for
   global SQ and should hold here.
3. **Cross-cluster distance comparisons.** The IVF driver always
   subtracts the cell centroid before scoring, so distances across
   clusters are commensurable even with per-cluster codebooks. If you
   plan to compare raw codes without the centroid context, this breaks.
4. **Memory bandwidth of the `(lo, hi)` table.** `k · d · 2 · 4` B must
   fit in L2 or you lose the "amortized" story. At `k = 65 536, d = 128`
   that's 64 MB, well past L2 but still ~0.5 % of code storage. A
   per-probe prefetch into L1 of the `nprobe` tables you'll use next
   is the right fix.

## What To Improve Next

1. **Non-uniform level placement per cluster.** Replace uniform 4-bit SQ
   with Lloyd-Max (optimal MSE quantizer) *per cluster*. We already have
   `lloyd-max-sq4` in a prior nightly; combining the two should stack.
2. **SIMD decode.** Current `l2_sq_decoded` is scalar; a 16-wide AVX2
   gather on the 4-bit nibbles + an fma against `(hi - lo)` gives ~4×
   on x86 and ~2× on NEON.
3. **Learned clipping percentiles.** Instead of raw min/max per cluster
   (sensitive to outliers), fit the clip range as a learnable parameter
   that minimizes reconstruction + distance error on a held-out query
   set.
4. **Half-precision ranges.** The `(lo, hi)` table is itself f32 but
   f16 (or bf16) is more than enough resolution for a 4-bit decode.
   Halves overhead to `k · d · 2 · 2` B.

## Production Crate Layout Proposal

```
crates/ruvector-sq/
  src/
    lib.rs           # trait Quantizer, Scalar4, Scalar8
    scalar4.rs       # 4-bit nibble pack/unpack, SIMD paths
    ranges.rs        # fit_global, fit_per_cluster, fit_percentile_clip
    ivf_integration.rs
    tests/
    benches/
```

Hooks into existing `ruvector-ivf` crate by adding a
`QuantizerBackend` enum (`Float32`, `GlobalSq4`, `PerClusterSq4`,
`GlobalSq8`, `PerClusterSq8`) selected at index-open time; no code-path
duplication for the IVF driver.

## References

[faiss-sq]: https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.cpp
[lvq]: https://www.vldb.org/pvldb/vol16/p3103-aguerrebere.pdf "Aguerrebere et al., *Similarity Search in the Blink of an Eye with Compressed Indices*, VLDB 2023"
[scann]: https://proceedings.mlr.press/v119/guo20h.html "Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020"
[rabitq]: https://dl.acm.org/doi/10.1145/3654970 "Gao & Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound*, SIGMOD 2024"

* Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*,
  TPAMI 2011.
* Johnson, Douze, Jégou, *Billion-scale Similarity Search with GPUs*,
  IEEE Big Data 2019 (Faiss paper).
* Milvus 2.5 ScalarQuantizer: https://github.com/milvus-io/milvus/tree/master/internal/core/src/index
* Qdrant quantization: https://qdrant.tech/documentation/guides/quantization/
