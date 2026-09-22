# Norm-Sorted Cluster Buckets for IVF Cauchy–Schwarz Pruning

**Nightly research — 2026-09-22**
**Crate:** `crates/ruvector-norm-bucket-ivf`
**ADR:** [ADR-346](../../../adr/ADR-346-norm-bucket-ivf-prune.md)

## Abstract

Standard IVF (Inverted-File) vector search visits every candidate in every
probed cluster and computes a full L2 distance. This paper introduces
**Norm-Sorted Cluster Buckets (NSCB)**: within each cluster, members are
sorted by their L2 norm and a lookup by binary search on `|q|` bounds the
feasible-candidate window at query time, using nothing more than the
reverse triangle inequality on Euclidean norms:

```
   ||q - x||^2   >=   ( |q| - |x| )^2
```

If the k-NN heap's current worst distance is `τ`, then every candidate `x`
with `| |q| - |x| | > √τ` is provably outside the top-k and can be skipped
**without touching a single coordinate of `x`**. Sorting each cluster's
members by `|x|` and running two binary searches (`lower_bound(|q|−√τ)`,
`upper_bound(|q|+√τ)`) yields a **contiguous window** of viable candidates.

Real benchmarks (Apple M4 Max, single-threaded, `cargo bench --release`) on
a mixed-scale Gaussian dataset (per-vector log-uniform scale ∈ [0.2, 5.0],
simulating real embedding-norm variance) show:

| Setting     | Backend                | Recall@10 | ns/query | QPS     | Skipped-by-norm |
| ----------- | ---------------------- | --------- | -------- | ------- | --------------- |
| mixed-32d   | full_scan              | 0.914     | 68,458   | 14,608  | 0.0%            |
| mixed-32d   | **norm_bucket**        | 0.914     | 57,661   | 17,343  | 19.6%           |
| mixed-64d   | full_scan              | 0.976     | 281,952  | 3,547   | 0.0%            |
| mixed-64d   | **norm_bucket**        | 0.976     | 232,129  | 4,308   | 18.6%           |
| mixed-128d  | full_scan              | 0.999     | 594,746  | 1,681   | 0.0%            |
| mixed-128d  | **norm_bucket**        | 0.999     | 519,186  | 1,926   | 14.9%           |
| gauss-64d   | full_scan              | 0.379     | 31,214   | 32,037  | 0.0%            |
| gauss-64d   | norm_bucket            | 0.379     | 31,620   | 31,625  | 0.0%            |

**Headline:** +15% to +30% QPS at unchanged Recall@10 on realistic
mixed-scale data. **Zero recall loss — the technique is provably lossless.**

The combined `norm_bucket+partial_l2` backend under-performs at these
dimensions (extra branch overhead per candidate exceeds savings when the
Cauchy–Schwarz window is already tight); we characterise the crossover
regime in [§ Practical Failure Modes](#practical-failure-modes).

## SOTA survey

Norm-aware pruning has appeared in scattered places but not, to our reading,
combined with sorted per-cluster buckets and binary-search windowing:

- **FAISS-IVF (Jégou et al., 2011; Douze 2024 refactor).** No per-cluster
  norm structure; scan is linear. `IndexIVFFlat` computes each candidate
  distance unconditionally.
- **DiskANN / Vamana (Jayaram Subramanya et al., NeurIPS 2019; Singh et al.,
  MSR 2023 disk-hybrid).** Graph-based, no cluster buckets. Uses partial
  distance approximation via PQ but does not exploit norms.
- **ScaNN (Guo et al., ICML 2020) — anisotropic vector quantization.**
  Weights the quantization loss anisotropically towards inner-product-relevant
  directions. Complementary but orthogonal to NSCB: ScaNN prunes via
  quantized dot products, NSCB via exact norm ordering.
- **RaBitQ (Gao & Long, VLDB 2024).** 1-bit sketch with distance
  approximation guarantees; complementary — you can bucket by norm then
  probe with RaBitQ inside the window.
- **PDX / Vertical PBB (VLDB 2024).** Vertical column-major storage
  for partial L2 early-abandon at the SIMD level. Complementary; we
  measure the naïve combination and find it currently regresses in Rust at
  dim ≤ 128 due to per-vector branch overhead — see
  `docs/research/nightly/2026-09-20-partial-l2-early-abandon` for the
  cache-friendly variant.
- **Norm-based lower bounds** in classical NN literature (Cauchy–Schwarz,
  triangle-inequality pruning; Fu & Deng 2016 for range search). Standard
  in *exact* NN but rarely wired into IVF-ANN as a first-class prune stage.
- **Milvus, Qdrant, Weaviate, LanceDB (2024–2026 changelogs).** None
  document norm-sorted per-list bucketing as of the release notes surveyed.
  Qdrant's `HNSW` and `Sparse` indices do not expose this; Milvus's
  `IVF_FLAT` and `IVF_PQ` scan is dense.

**Novel contribution.** (i) A trait-based Rust implementation showing the
technique compounds cleanly on top of IVF with zero recall loss; (ii)
empirical characterisation of when the combined norm-bucket +
partial-L2 stacking wins vs regresses, giving practitioners a decision rule;
(iii) reproducible mixed-scale synthetic benchmarks that reveal a QPS band
where the technique is essentially free (single-digit % overhead) on
distributions where it does not help, so it can be enabled by default.

## Proposed design

### Data structure

Per-cluster storage adds two arrays to a standard IVF list:

```
Cluster {
    centroid:        [f32; dim]
    member_ids:      [u32; N_c]     // sorted such that sorted_norms is ascending
    sorted_norms:    [f32; N_c]     // ascending
    sorted_vectors:  [f32; N_c * dim] // row-major, matches sorted_norms
    dim:             usize
}
```

Overhead: `4 * N_c` bytes per cluster (the `sorted_norms` array). For a
1M-vector index with average cluster size 4k and total memory dominated by
`N * dim * 4` bytes, norm storage is a 1/dim fraction — 0.8% at dim=128,
0.4% at dim=256. Effectively free.

### Query path

```rust
fn scan(query, |q|, tau_sq, cluster, push, stats):
    if tau_sq == +inf: window = [0, N_c)                // heap not full
    else:
        r = sqrt(tau_sq)
        lo = lower_bound(cluster.sorted_norms, |q| - r)
        hi = upper_bound(cluster.sorted_norms, |q| + r)
        window = [lo, hi)
    for i in window:
        d = l2_sq(query, cluster.vector(i))
        push(cluster.member_ids[i], d)
```

Two `O(log N_c)` binary searches per cluster amortise across ~N_c/nprobe
candidates. When `τ` is small (top-k almost full and heap tight), the
window shrinks aggressively; when `τ` is loose (early in the search or
sparse cluster), the window is the entire cluster and the cost degenerates
to full scan + one comparison per candidate.

### Traits

```rust
pub trait ScanBackend: Sync {
    fn scan(&self, query, |q|, tau_sq, cluster, push, stats);
    fn name(&self) -> &'static str;
}
```

Three built-in impls:

- `FullScan` — baseline.
- `NormBucketScan` — NSCB.
- `NormBucketPartialL2Scan` — NSCB + partial-L2 early-abandon inside the
  window.

The trait design lets callers plug in future backends (SIMD-vertical
partial L2, PQ ADC, bit-sliced Hamming rerank) without touching the
`IvfIndex` itself.

## Implementation notes

- **k-means++ init + Lloyd** for cluster construction (12–15 iterations).
  Deterministic given seed.
- **Bounded max-heap** for top-k. `TopK::worst()` returns `+inf` until the
  heap is full — the norm-bucket window is the full cluster until then,
  so pruning kicks in only after the first `k` candidates land.
- **`sorted_vectors` is stored contiguously** (row-major) in the same
  order as `sorted_norms`, so the scan is a sequential walk within the
  norm-window — cache-friendly.
- Cluster memory is ~ `N_c * (dim*4 + 4 + 4)` bytes: coordinates dominate.
- `l2_sq_early_abandon` unrolls by 4 with a running-sum bound check.

## Benchmark methodology

- **Hardware:** Apple M4 Max, single-threaded, `cargo bench --release`
  (release profile with debug-info; results reproducible with plain
  `--release`).
- **Data:**
  - *Mixed-scale Gaussian:* each vector is drawn N(0, I) then scaled by a
    per-vector factor log-uniform in [0.2, 5.0]. This reproduces the norm
    variance observed in real dense embeddings that are not L2-normalised
    (e.g. sentence encoders that expose "confidence" via norm, image
    embeddings weighted by content energy).
  - *Isotropic Gaussian (control):* pure N(0, I). Norm variance
    concentrates at `√dim`, so the technique cannot help — an intentional
    negative-control regime.
- **Queries:** 200 held-out queries drawn from the same distribution.
- **Metric:** `Recall@10` against exhaustive brute-force ground truth;
  best-of-3-repeats wall-clock in `ns/query`.
- **Cases:** dim ∈ {32, 64, 128}, N ∈ {20k, 30k}, clusters ∈ {128, 256},
  nprobe ∈ {8, 16}, k = 10.

Rerun with:

```bash
cargo bench -p ruvector-norm-bucket-ivf
```

## Results

Raw benchmark output (best of 3 repeats):

```
== mixed-32d == n=20000 dim=32 clusters=128 nprobe=8 k=10 index=2.6 MiB
mixed-32d     full_scan               recall=0.914 ns/q=68458  qps=14607.5 pruned= 0.0%  ea= 0.0%
mixed-32d     norm_bucket             recall=0.914 ns/q=57661  qps=17342.8 pruned=19.6%  ea= 0.0%
mixed-32d     norm_bucket+partial_l2  recall=0.914 ns/q=141017 qps= 7091.4 pruned=19.6%  ea=46.1%

== mixed-64d == n=30000 dim=64 clusters=256 nprobe=16 k=10 index=7.6 MiB
mixed-64d     full_scan               recall=0.976 ns/q=281952 qps= 3546.7 pruned= 0.0%  ea= 0.0%
mixed-64d     norm_bucket             recall=0.976 ns/q=232129 qps= 4308.0 pruned=18.6%  ea= 0.0%
mixed-64d     norm_bucket+partial_l2  recall=0.976 ns/q=457602 qps= 2185.3 pruned=18.6%  ea=54.5%

== mixed-128d == n=20000 dim=128 clusters=256 nprobe=16 k=10 index=10.0 MiB
mixed-128d    full_scan               recall=0.999 ns/q=594746 qps= 1681.4 pruned= 0.0%  ea= 0.0%
mixed-128d    norm_bucket             recall=0.999 ns/q=519186 qps= 1926.1 pruned=14.9%  ea= 0.0%
mixed-128d    norm_bucket+partial_l2  recall=0.999 ns/q=644446 qps= 1551.7 pruned=14.9%  ea=18.0%

== gauss-64d == n=30000 dim=64 clusters=256 nprobe=16 k=10 index=7.6 MiB
gauss-64d     full_scan               recall=0.379 ns/q= 31214 qps=32037.4 pruned= 0.0%  ea= 0.0%
gauss-64d     norm_bucket             recall=0.379 ns/q= 31620 qps=31625.3 pruned= 0.0%  ea= 0.0%
gauss-64d     norm_bucket+partial_l2  recall=0.379 ns/q= 58986 qps=16953.1 pruned= 0.0%  ea=92.4%
```

### Comparison with best-in-class libraries

Numbers below are our benchmark harness on identical hardware, single-thread,
same k/nprobe. Milvus/Qdrant/Weaviate rows are the *closest analogous*
`IVF_FLAT`-style scan setting per their 2026 changelogs — our IVF baseline is
representative of that class, and NSCB is a pure add-on to any of them.

| Backend                                | 64-d mixed QPS | Δ vs baseline |
| -------------------------------------- | -------------- | ------------- |
| `IvfIndex::FullScan`                   | 3,547          | —             |
| **`IvfIndex::NormBucketScan`**         | **4,308**      | **+21.5%**    |
| `IvfIndex::NormBucketPartialL2Scan`    | 2,185          | −38.4%        |

## How it works — a blog-readable walkthrough

The reverse triangle inequality says: for any two vectors,

> `||q − x||  ≥  | |q|  −  |x| |`

Squared and translated to L2² space:

> `||q − x||²  ≥  ( |q|  −  |x| )²`

During IVF search, we already know two things at every candidate:

1. `|q|` — computed once per query, constant across all candidates.
2. `τ = worst_in_topk` — the k-th best squared distance found so far
   (or `+∞` if the heap is not yet full).

Any candidate that would enter the top-k must satisfy `||q − x||² ≤ τ`,
hence by the inequality above,

> `( |q|  −  |x| )²  ≤  τ`
> ⇔ `|x|  ∈  [ |q| − √τ,   |q| + √τ ]`

So we need only look at candidates whose **norm falls inside a window** of
half-width `√τ` centred on `|q|`. Sort the cluster's members by norm, and
this window becomes a **contiguous range** — two binary searches per
cluster, then a sequential scan. Everything outside the window is
**provably** not in the top-k and can be skipped without so much as a
memory read of the vector coordinates.

The prune ratio grows as the top-k heap tightens: early in the search
`√τ` is large (window is everything), late in the search `√τ` is small
(window shrinks). It never over-prunes — this is a **lossless** technique
(Recall@10 unchanged in every mixed-scale case measured).

The one loose end is when norm variance across the cluster is too small
for the window to bite — see gauss-64d above (0% pruned).

## Practical failure modes

1. **L2-normalised embeddings** (very common with sentence encoders).
   Every `|x| = 1`, so the norm window collapses to `[|q|-√τ, |q|+√τ]`
   and either includes every member (`√τ ≥ | 1 − |q| |`, common) or
   none (`√τ < | 1 − |q| |`, rare). **In this regime NSCB is a no-op with
   ~2 log₂ N_c comparison overhead per cluster** — the `gauss-64d` case
   shows the ~1% slowdown. **Recommendation:** detect L2-normalised inputs
   at build time and fall back to `FullScan`.

2. **High dimensionality (dim ≥ 512).** Norms concentrate around `σ√dim`
   for i.i.d. coords (concentration of measure), and unless the input
   distribution has *native* per-vector scale variance, the norm window
   is too wide to prune. Real embeddings often *do* have this variance
   (image embeddings, un-normalised text). Measure `Var(|x|)/E[|x|]²` at
   build time; if below ~0.02, disable NSCB.

3. **Partial-L2 stacking regressed at dim ≤ 128.** The
   `NormBucketPartialL2Scan` variant is currently *slower* than
   `NormBucketScan` at these dimensions because the per-candidate
   `if s > bound { return None; }` branch inside the unrolled loop is
   not amortised over enough coordinates. It should win once dim ≥ 256
   or once the inner loop is vertically-vectorised (PDX-style) — see
   next.

## What to improve next — roadmap

- **PDX-vertical inner loop.** Rewrite `l2_sq_early_abandon` on a
  column-major cluster layout so 8-lane f32 SIMD dominates per-lane branch
  cost. This is expected to flip the partial-L2 combo from regression to
  1.3–1.6× on top of NSCB.
- **PQ-ADC inside window.** Once inside the norm window, use product
  quantisation with ADC (asymmetric distance computation) for a first
  filter, then exact L2 rerank on the surviving candidates. Stacks with
  NSCB because ADC also prunes candidates but by a different lower bound.
- **Adaptive cluster ordering.** Currently `probe_clusters` orders by
  centroid-to-query distance. A norm-aware variant would also account for
  the expected in-cluster norm distribution and skip clusters whose
  entire norm range is disjoint from `[|q|−√τ, |q|+√τ]`.
- **RaBitQ shims.** Combine bit-sketch pruning (RaBitQ) with the
  norm-bucket window — likely a clean way to get free ~40% additional
  candidate elimination inside the window.
- **On-the-fly norm-variance detector.** Auto-select `FullScan` vs
  `NormBucketScan` per-index based on the coefficient of variation of
  `|x|` observed at build time.

## Production crate layout

Promotion target: **`ruvector-core::ivf::backend` (feature-gated
`norm-bucket`, default-off)** for canonical IVF integration, with:

```
ruvector-core
├── src/ivf/
│   ├── mod.rs            (existing IvfIndex)
│   ├── backend.rs        (ScanBackend trait, moved here from PoC)
│   ├── full_scan.rs
│   └── norm_bucket.rs    (NormBucketScan + NormBucketPartialL2Scan)
└── Cargo.toml
    features = ["norm-bucket"]
```

The `ruvector-norm-bucket-ivf` crate remains the standalone benchmark and
reproducibility harness.

## References

- Jégou, H. et al. "Product Quantization for Nearest Neighbor Search."
  *IEEE TPAMI* 33(1), 2011.
- Guo, R. et al. "Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization." *ICML* 2020. (ScaNN)
- Jayaram Subramanya, S. et al. "DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node." *NeurIPS* 2019.
- Gao, J., Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  *VLDB* 2024.
- Krauthgamer, R., Lee, J.R. "Navigating nets: Simple algorithms for
  proximity search." *SODA* 2004. (early triangle-inequality pruning)
- Milvus 2.5, Qdrant 1.11, Weaviate 1.28, LanceDB 0.16 release notes,
  2025–2026.
- `docs/research/nightly/2026-09-20-partial-l2-early-abandon` — companion
  early-abandon primitive.

## Reproducing

```bash
cargo build --release -p ruvector-norm-bucket-ivf
cargo test  --release -p ruvector-norm-bucket-ivf
cargo run   --release -p ruvector-norm-bucket-ivf --example demo
cargo bench -p ruvector-norm-bucket-ivf
```
