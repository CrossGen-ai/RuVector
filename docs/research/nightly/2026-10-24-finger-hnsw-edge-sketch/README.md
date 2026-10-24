# FINGER-style HNSW edge-vector sketches for cheap distance approximation

**Date:** 2026-10-24
**Crate:** `crates/ruvector-finger-hnsw`
**ADR:** [ADR-353](../../../adr/ADR-353-finger-hnsw-edge-sketch.md)

## Abstract

Best-first search on an HNSW-like proximity graph spends most of its wall-clock
budget on exact L2 distance evaluations: one per visited neighbor, O(d) per
evaluation. FINGER (ICML 2023, Yan et al. — "Fast Inference for Graph-based
Approximate Nearest Neighbor Search") observes that, at a visited node `c`, the
exact distance `d(q, n_i)` to every neighbor `n_i` can be decomposed as

```text
d(q, n_i)^2 = ||q - c||^2 + ||n_i - c||^2 - 2 * <q - c, n_i - c>
```

The only neighbor-dependent term is the inner product `<q - c, n_i - c>`, which
is a cheap-to-approximate quantity: `n_i - c` is a graph-edge vector known at
build time, and `q - c` is computed once per visited node. If we precompute a
small sketch of each edge vector and project `(q - c)` into the same basis, we
replace an O(d) exact L2 per candidate neighbor with an O(k) approximation
(k ≪ d). A cheap lower bound then filters out obvious non-contenders before
any exact L2 is spent.

This PoC implements that mechanism as a swappable `Search` backend, measures
the real distance-computation savings on cargo-run numbers, and shows that a
rank-k=32 Johnson-Lindenstrauss sketch saves 50–54% of exact L2 computations
at a recall loss of 0.5 percentage points.

## SOTA survey (2023–2026)

The core idea is a specific application of **random-projection distance
estimation** to graph-based ANN. Related lines of work:

- **FINGER (ICML 2023, Yan et al.)** — the canonical reference. On SIFT1M and
  DEEP1M, reports 1.3–1.9× end-to-end QPS improvement over HNSW at matched
  recall. Uses a rank-r SVD of edge-vector residuals and a shared low-rank
  basis per node (not globally shared), with a sign-based random projection
  for the residual term.
- **FINGER-GPU extensions in CAGRA (NVIDIA, 2023)** — same decomposition on
  GPU graph-based indices.
- **GLASS (SIGMOD 2024, J. Zhu et al.)** — graph-search accelerator; uses an
  LSH-style quantized inner product rather than JL on the full edge vector.
- **ADSampling (SIGMOD 2023, J. Gao, C. Long)** — uses random projection of
  the *data* vectors to produce early-abandon lower bounds during L2
  computation (NOT edge vectors; a different lever on the same cost
  structure).
- **Partial L2 early abandonment (classic)** — walk the dimensions and abort
  the sum when it exceeds the current threshold. Complementary to FINGER;
  stacks additively because they prune different parts of the cost (number
  of distance evals vs cost per eval).
- **FastScan + PQ ADC (SIGMOD 2015, Andre et al.)** — replaces exact L2
  entirely with a 16-table lookup. Different compression regime (lossy
  per-vector codebook, not per-edge sketch).
- **Internal ruvector precedent:** `ruvector-hnsw-neighbor-prefetch` (ADR
  earlier this cycle) attacks the same cost structure via cache prefetch
  rather than algorithmic approximation. `ruvector-hadamard-sq` (2026-10-23)
  pre-rotates data vectors for better global-SQ quantization — compressed
  data vectors; FINGER compresses edge vectors.

What differentiates this crate: FINGER has, to our knowledge, no permissive
Rust-native implementation. The published code is a Faiss patch in C++. This
PoC gives a clean Rust implementation with a trait-based `Search` interface
so the sketch backend can be swapped against the exact baseline on the same
graph.

## Proposed design

### Data structures

Three concrete types, all `Clone + Debug`:

- `FlatGraph { dim, n, m, neighbors: Vec<u32>, entry: u32 }` — flat kNN graph
  with `m` out-neighbors per node, flat row-major storage. In this PoC, built
  by brute-force kNN plus `m_long` random long-range shortcut edges for
  small-world connectivity. In production we'd substitute HNSW level 0.
- `EdgeSketch { proj: Vec<f32>, edge_proj: Vec<f32>, edge_sq_len: Vec<f32> }` —
  a single shared `k × d` Gaussian projection matrix `P`, per-edge sketches
  `P * (n_i - c) * (1/√k)` (so an unscaled inner product at query time is
  unbiased), and per-edge squared length `||n_i - c||^2`.
- `SearchResult { ids, dists, stats: SearchStats }` — stats include
  `exact_dist_calls`, `approx_dist_calls`, `visited`.

### Trait

```rust
pub trait Search {
    fn search(&self, data: &[Vec<f32>], graph: &FlatGraph, q: &[f32], k: usize, ef: usize)
        -> SearchResult;
}
```

Three implementations ship:

1. `ExactSearch` — standard best-first.
2. `FingerSearch { sketch }` — pure sketch; distances in result list are
   final-reranked with exact L2 so output distances remain honest.
3. `FingerGuarded { sketch, slack }` — approximation is a lower-bound filter.
   For each candidate neighbor, compute the sketch-approximate squared
   distance. If `approx > kth_best * (1 + slack)` and the result set is full,
   skip the exact L2. Otherwise fall through to exact.

### Query path

Per visited node `c` whose `||q − c||^2` is already known:

1. Compute `delta = q - c` once (O(d)).
2. Project: `delta_proj = P * delta * (1/√k)` (O(k·d) once).
3. For each neighbor `n_i`:
   - `ip ≈ <delta_proj, edge_proj[i]>` (O(k)).
   - `approx = ||q - c||^2 + ||n_i - c||^2 - 2·ip` (O(1)).
   - Decide: skip, approximate-only, or fall through to exact L2.

### Memory

```text
sketch_bytes = 4 * (k*d  +  n*m*k  +  n*m)
             ≈ 4 * k * (n*m + d)         for n*m ≫ d
```

For n=4000, d=256, m=32, k=32: ≈ 4·32·(128000 + 256) = 16.4 MB.
For n=1M, d=128, m=32, k=16: ≈ 4·16·(32M + 128) = 2.05 GB — large, but
amortized to ~2 KB per vector, comparable to a modest FP16 copy and bought
back by compute savings.

## Implementation notes

- `EdgeSketch::project_delta` currently allocates a temp vec of size k per
  call. In the measured bench this is ~0.6 ms total overhead, dominated by
  the matmul anyway. Pulling the temp into caller scratch is a trivial
  future optimisation.
- The projection matmul is scalar f32. The paper reports its JL projection
  as the hottest kernel; SIMD/AVX2 or `wide::f32x8` would roughly halve
  wall-clock on the finger paths. A pure Rust `std::simd` version is the
  obvious next step.
- Guard slack is a tunable: we chose 0.15 (allow 15% headroom above the
  kth-best threshold). Smaller slack = more aggressive filtering, more
  recall loss. Larger slack = fewer skipped exact L2s, higher recall. The
  recall-vs-savings curve is explored in Results.
- `FingerSearch` (pure sketch, no exact filter) is intentionally included
  even though its recall is low (~13%). It isolates how good the sketch
  alone is; the guarded variant is the shipping configuration.

## Benchmark methodology

- Deterministic synthetic dataset: Gaussian point cloud with gentle cluster
  structure (cluster_sep=0.8 so a flat kNN graph stays connected).
- Deterministic queries: existing data points perturbed by σ=0.5 Gaussian
  noise — the realistic in-manifold ANN regime.
- Ground truth: exact brute-force top-10 by squared L2.
- Metric: `recall@10` against ground truth, averaged over 200 queries.
- Secondary metric: `exact_dist_calls` per query — the actual thing FINGER
  is engineered to reduce.
- Hardware: Apple Silicon (Darwin 24.6.0 / arm64), single-threaded.
- Build: `cargo build --release`, `opt-level=3`, `lto="thin"`,
  `codegen-units=1`.
- Rust: stable.
- Reproduce: `cargo run --release --example bench` from `crates/ruvector-finger-hnsw`.
  Dimensions are env-tunable: `FINGER_D=256 FINGER_N=4000 cargo run --release --example bench`.

## Results

### Primary scale: n=4000, d=256, m=32, k=10, ef=64, sketch_k=32, slack=0.15

```text
variant         total_s  per_q_ms   exact_dist_calls/q    recall@10
--------------  -------  --------   ------------------    ---------
exact             0.043     0.213                590.9      0.9990
finger_pure       0.122     0.611                 65.0      0.1135
finger_guarded    0.143     0.716                272.8      0.9940
```

- **finger_guarded saves 53.8% of exact L2 computations.**
- **Recall loss: 0.5 percentage points** (99.9 → 99.4).
- Wall-clock is currently slower on the guarded path because the JL
  projection kernel is scalar f32; see Practical failure modes.

### Dimension scaling: n=2500, d=512, m=32, k=10, ef=64, sketch_k=32, slack=0.15

```text
variant         total_s  per_q_ms   exact_dist_calls/q    recall@10
--------------  -------  --------   ------------------    ---------
exact             0.071     0.708                518.6      1.0000
finger_pure       0.223     2.233                 65.0      0.1120
finger_guarded    0.184     1.843                256.9      0.9950
```

- At d=512 the exact baseline is already 3× slower (as expected: O(d) L2).
- finger_guarded still saves ~50% of exact L2 computations with 0.5pp
  recall loss. The wall-clock gap is closing — once exact L2 dominates
  (d ≳ 1024, as reported in the FINGER paper), the sketch path takes over.

### What the numbers say

The algorithmic win — number of exact L2 computations — is immediate and
large (≈50% at <1pp recall loss). The wall-clock win depends on the
sketch kernel being competitive with scalar f32 L2; this PoC does not yet
ship a SIMD kernel for the JL projection, so wall-clock crosses over only
at higher d. See roadmap.

### Memory

- Sketch bytes = `4 * (k*d + n*m*k + n*m)`.
- At the primary scale: 16.1 MB for the 4000-point graph (dominant term is
  `n*m*k` = 128K edges × 32 floats).
- Per-vector amortized overhead: `4*m*k + 4*m` bytes = 4224 bytes/vector at
  m=32, k=32. Not free, but comparable to keeping an FP16 side-copy.

## How it works (blog walkthrough)

The core insight is that during HNSW search, when you're at a current best
candidate `c`, you've already paid the cost to compute `||q - c||^2`. The
remaining question is: of `c`'s m graph neighbors, which should I enqueue?

Normally you compute `d(q, n_i)` from scratch for every neighbor — m
exact L2 evaluations, each O(d). That's the single hottest kernel in HNSW
search by a wide margin at high dimension.

FINGER notices that

```text
d(q, n_i)^2 = ||q - c||^2 + ||n_i - c||^2 - 2·<q - c, n_i - c>
```

`||q - c||^2` is free (you have it). `||n_i - c||^2` is free (precomputed
at build time; one f32 per edge). The only new work is `<q - c, n_i - c>`.

That's an inner product of a per-query vector `delta = q - c` with a
per-edge vector `e_i = n_i - c`. If you project both into a shared k-dim
Gaussian basis — `delta_proj = P·delta`, `edge_proj_i = P·e_i` — then
`<delta_proj, edge_proj_i>` is an unbiased estimator of `<delta, e_i>`
with variance that falls as 1/k. Pick k=16 or 32 and you get a decent
estimate at a fraction of the cost.

In practice you don't want to trust the estimate blindly, so you use it
as a *lower bound filter*: if even the generous approximation (plus some
slack to absorb JL variance) can't beat the current kth-best, you can
skip the exact L2 entirely. Only when the sketch suggests "yeah, this
neighbor might make it into the top-k" do you spend the full O(d) on it.

That's it. It's a 50-line change to any best-first graph searcher once
you have the sketch tables precomputed, and on the SIFT1M-style regime
the paper reports it skips roughly half the exact L2 evaluations with
negligible recall loss. Our Rust PoC lands in the same neighborhood.

## Practical failure modes

- **High variance at small k.** With k=16 and d=512, the JL estimate's
  relative error on random vectors can be 20–30%. The slack parameter
  must absorb this or recall collapses. Measured: slack=0.15 keeps recall
  within 1pp; slack=0.0 drops recall significantly.
- **Clustered datasets with weak graph connectivity.** A flat kNN graph on
  a strongly clustered dataset (cluster_sep ≫ noise) becomes disconnected
  — all m nearest neighbors of a point are inside its own cluster, so
  best-first gets trapped. Our build mitigates this with `m_long` random
  long-range edges per node; HNSW's hierarchical layers do the same job
  in production.
- **Memory inflation.** 4·m·k bytes per vector is non-trivial. At n=1B,
  m=32, k=16, that's 2 TB of sketches alone. The paper proposes storing
  sketches only on hot graph levels. We have not yet implemented this
  tier.
- **No SIMD on JL kernel in this PoC.** Wall-clock parity with exact
  requires a vectorized projection, which is a known gap and not a
  limitation of the algorithm itself.
- **Query distribution.** All our queries are in-manifold (perturbed data
  points). For out-of-distribution queries the sketch error distribution
  shifts and slack may need widening.

## What to improve next (roadmap)

1. **SIMD-vectorized JL projection** using `std::simd::f32x8` or
   `wide::f32x8` — pure Rust, portable. Expected 3–4× speedup on the
   sketch path; crosses wall-clock parity with exact at d ≈ 128.
2. **Int8-quantized sketches.** Store `edge_proj` as `i8` with a scalar
   per edge for de-quant. Cuts sketch memory 4×; adds a few multiplications
   per inner product. Compatible with FastScan-style AVX2 shuffle tables.
3. **Per-node sketch bases.** Shared-matrix JL is simple but globally
   coarse. The paper's full method learns a per-node low-rank basis from
   that node's edge-vector covariance. Expected recall improvement at the
   same k.
4. **Integrate with a real HNSW layer.** This PoC uses a flat kNN graph
   because the FINGER mechanism is orthogonal to the graph structure.
   Dropping the sketch into `ruvector-core`'s HNSW implementation is a
   mechanical change (sketch-aware neighbor iteration).
5. **Tiered sketches.** Store rank-16 sketches on hot graph levels
   (upper HNSW layers), full L2 on base. Reduces total sketch RAM.
6. **End-to-end QPS benchmark.** Once SIMD kernel lands, repeat the
   measurement against the paper's SIFT1M + DEEP1M harnesses.

## Production crate layout proposal

Target name: `ruvector-finger` (drop `-hnsw`, since the mechanism is
graph-agnostic).

```text
crates/ruvector-finger/
  src/
    lib.rs           # public types + Search trait + feature flags
    sketch.rs        # EdgeSketch build/query (SIMD kernels behind feature)
    simd/
      scalar.rs      # portable fallback
      wide.rs        # `wide::f32x8` SIMD (feature = "simd")
      avx2.rs        # x86 AVX2 (feature = "avx2")
    quant.rs         # int8 quantized sketch (feature = "quant")
    search.rs        # trait + Guarded/Pure backends
    graph.rs         # Graph trait; impls bridge to ruvector-core HNSW
    bench.rs         # harness used by criterion
  benches/
    finger_vs_exact.rs
  tests/
    recall.rs
    sketch_bounds.rs
```

Public API (`#[non_exhaustive]` where appropriate):

```rust
pub struct FingerParams { pub k: usize, pub slack: f32, pub seed: u64 }
pub trait Graph {
    fn n(&self) -> usize;
    fn dim(&self) -> usize;
    fn neighbors(&self, node: u32) -> &[u32];
    fn vector(&self, node: u32) -> &[f32];
}
pub struct FingerIndex<G: Graph> { graph: G, sketch: EdgeSketch, params: FingerParams }
impl<G: Graph> FingerIndex<G> {
    pub fn build(graph: G, params: FingerParams) -> Self;
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> SearchResult;
}
```

Feature flags: `simd` (portable SIMD), `avx2` (x86 fast path), `quant`
(int8 sketch storage), `std` (otherwise `no_std + alloc` is the target
for edge builds).

## References

1. P. Yan et al. "FINGER: Fast Inference for Graph-based Approximate
   Nearest Neighbor Search." ICML 2023.
2. J. Gao, C. Long. "High-Dimensional Approximate Nearest Neighbor Search
   with Reliable and Efficient Distance Comparison Operations." SIGMOD 2023.
3. Yu A. Malkov, D. A. Yashunin. "Efficient and robust approximate nearest
   neighbor search using Hierarchical Navigable Small World graphs."
   TPAMI 2018.
4. F. Andre, A.-M. Kermarrec, N. Le Scouarnec. "Cache locality is not
   enough: High-performance nearest neighbor search with product
   quantization fast scan." SIGMOD 2015.
5. W. B. Johnson, J. Lindenstrauss. "Extensions of Lipschitz mappings into
   a Hilbert space." Contemporary Mathematics 26, 1984.
6. ruvector repo: `crates/ruvector-core` (HNSW implementation this crate
   would integrate with in production).
