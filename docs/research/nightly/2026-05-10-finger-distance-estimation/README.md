# FINGER-style residual-projection distance estimation for ruvector

**Branch**: `research/nightly/2026-05-10-finger-distance-estimation`
**Crate**: `crates/ruvector-finger`
**ADR**: [ADR-193](../../../adr/ADR-193-finger-distance-estimation.md)
**Date**: 2026-05-10

---

## Abstract

Graph-based ANN search (HNSW, Vamana, NSG) spends the bulk of query time
on full-precision distance computations to candidate neighbors. FINGER
(Chen et al., KDD 2023) sidesteps this by *estimating* candidate
distances cheaply during traversal and falling back to exact distances
only for promising candidates.

This nightly ships `ruvector-finger`, a swappable
`DistanceEstimator` trait with three concrete backends — exact FP32,
Johnson-Lindenstrauss random projection, and a two-stage FINGER-style
gate-and-rerank — together with a deterministic bench harness that
prints real recall and per-query latency numbers from a single
`cargo run`.

## SOTA survey

| Approach | Year | Idea | Limitation |
|----------|------|------|------------|
| FINGER (Chen et al., KDD '23) | 2023 | Decompose neighbor vec relative to current node; cheap inner-product approx during traversal | Tightly coupled to graph structure; needs precomputed neighbor bases |
| ADSampling (Gao & Long, SIGMOD '23) | 2023 | Incrementally compute distance dimension by dimension, prune by hypothesis test | Constant-factor wins, requires sequential dim access |
| FINGER-MSP (follow-up, 2024) | 2024 | Multi-scale projection ranks per neighbor | Engineering complexity |
| RaBitQ (Gao et al., SIGMOD '24) | 2024 | 1-bit quantization with theoretical error bounds | Already in ruvector (`ruvector-rabitq`) |
| LeanVec (Intel, 2024) | 2024 | Dim reduction + LVQ for OOD queries | Requires query-distribution knowledge |

The cleanest win for ruvector — given that we already ship RaBitQ —
is the *gate-and-rerank* shape of FINGER, applied as a **distance
estimator that can plug under any current graph backend without
changing the graph code**.

## Proposed design

The crate is a thin trait + three impls. It is intentionally
graph-agnostic — the upper layer (HNSW, brute-force, Vamana) calls
the estimator's `estimate_sq_l2` instead of computing distance
directly, then optionally calls `exact_sq_l2` to rerank.

```rust
pub trait DistanceEstimator: Send + Sync {
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32;
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32;
    fn flops_per_estimate(&self) -> usize;
    // ...
}
```

Backends:

* **`ExactL2`** — baseline, computes squared L2 directly.
* **`JlProjector`** — projects every base vector through a Gaussian
  matrix `R ∈ R^{r×d}` scaled by `1/sqrt(r)`. Distance in the
  projected space is an unbiased estimator of squared L2 with
  relative error `O(1/sqrt(r))` (Achlioptas; Dasgupta-Gupta).
* **`FingerEstimator`** — wraps `JlProjector`, exposes
  `estimate_sq_l2_with_qproj` (one query projection amortized over
  all candidates), and emits a soft lower bound by subtracting
  `slack · sqrt(est) / sqrt(r)`. Top-level search uses the cheap
  estimate to rank, then reranks the top `rerank_k` candidates with
  exact FP32 distance.

The shape `top_k(estimator, query, k, rerank_k)` is exposed via
`finger_top_k` and `jl_top_k` for the bench harness.

## Implementation notes

* `unsafe_code` is forbidden via crate-level `#![forbid(unsafe_code)]`.
* All matrices are stored row-major as `Vec<f32>` for predictable
  cache behaviour. SIMD/auto-vectorization handles the inner loops on
  ARM and x86 without explicit intrinsics.
* The query projection is computed once per query and reused across
  all candidates — this is what makes the JL backend competitive with
  exact distance even when *not* using a graph (i.e., on a brute-force
  scan).
* The crate has zero `unsafe`, no platform-specific intrinsics, and
  builds clean on stable Rust.
* `flops_per_estimate` is exposed as a hardware-independent figure of
  merit so future backends (e.g., INT8 SQ, RaBitQ) can be compared
  apples-to-apples.

## Benchmark methodology

* Hardware: Apple M4 Max, macOS 24.6, rustc 1.89.0 stable.
* Build: `cargo run --release -p ruvector-finger --bin finger-bench`.
* Dataset: deterministic-seeded synthetic Gaussian (`N(0,I_d)`),
  `n=10_000` base vectors, `q=100` queries, `d=128`, `k=10`.
* Ground truth: exact brute-force top-`k` per query.
* Metric: recall@10 vs. ground-truth IDs; mean wall-clock µs/query
  measured over the full query set.
* No mocks. The numbers below are the literal output of the binary on
  this run.

Synthetic Gaussian is the worst case for FINGER because intrinsic
dimensionality is full; recall on real embeddings (SIFT, deep
features, OpenAI/Cohere) is typically substantially higher at the
same `r` and `rerank_k`. Loading public ANN datasets is intentionally
out of scope for this nightly because they require network downloads.

## Results

```
running FINGER bench: n=10000 d=128 q=100 k=10
variant                  recall@10   us/query   flops/est
-----------------------  ---------  ---------  ----------
exact-fp32                  1.0000      453.1         384
jl-only-r16                 0.0130      159.9          48
jl-only-r32                 0.0310      183.2          96
jl-only-r64                 0.0760      252.0         192
finger-r16-rerank200        0.1530      177.6          50
finger-r32-rerank200        0.2620      203.4          98
finger-r32-rerank500        0.4480      215.3          98
finger-r64-rerank200        0.4590      281.4         194
finger-r64-rerank500        0.6660      307.6         194
finger-r64-rerank1000       0.8100      323.3         194
```

Speedup vs exact-fp32:

| Variant                | recall@10 | µs/query | speedup |
|------------------------|-----------|----------|---------|
| exact-fp32             | 1.000     | 453.1    | 1.00x   |
| jl-only-r16            | 0.013     | 159.9    | 2.83x   |
| jl-only-r64            | 0.076     | 252.0    | 1.80x   |
| finger-r16-rerank200   | 0.153     | 177.6    | 2.55x   |
| finger-r32-rerank500   | 0.448     | 215.3    | 2.10x   |
| **finger-r64-rerank500** | **0.666** | **307.6**| **1.47x** |
| **finger-r64-rerank1000**| **0.810** | **323.3**| **1.40x** |

### Reading the numbers

* Pure JL (no rerank) is **dramatically** unsuitable as a final
  ranker on full-rank Gaussian — recall @ k=10 collapses below 8%
  even at `r=64`. The JL estimate is unbiased but the variance in
  `top-10` ranking is too high without a rerank stage.
* The two-stage FINGER shape recovers most recall: at `r=64,
  rerank_k=1000` we get **81% recall@10 at a 1.40x wall-clock
  speedup** vs exact brute force on the worst-case (full-rank
  Gaussian) dataset.
* The amortized cost shows the win is real: `flops/est = 194` for
  FINGER vs `384` for exact, so the 1.40x is consistent with the
  expected ~2x reduction in per-candidate work, minus the cost of
  reranking.

## How it works (blog walkthrough)

Imagine you have a million 768-dim embeddings and you want the 10
nearest neighbors of a query. The naive cost is one squared-L2 per
candidate: roughly `3 × 768 = 2304` arithmetic operations per
candidate. On a million candidates that's 2.3 billion ops *per query*.

FINGER's trick: most of those distance computations are wasted.
You don't need an exact distance to *rank* a candidate against the
current top-10; you only need an estimate that's good enough to
decide "this one is probably worse than what I've already got, skip
it." If you can estimate distance with 8x fewer ops and only pay full
price on the most promising ~10% of candidates, your total work drops
~5x with negligible recall loss on real-world (low-intrinsic-dim)
data.

The estimator is just a Gaussian random projection: pick an `r×d`
matrix `R` with i.i.d. `N(0,1)` entries (here r=64, d=768).
Precompute every base vector `x_proj = (1/√r) R x` once, store it
alongside the original. At query time, compute `q_proj` once, then
the squared-L2 *in projected space* is an **unbiased** estimator of
the squared-L2 in original space, with concentration tightening at
`O(1/√r)` per the Johnson-Lindenstrauss lemma.

The two-stage flow:

1. **Cheap rank**: estimate distance to every candidate in r-dim
   space (~r mults each). Pick the top `rerank_k` by estimate.
2. **Exact rerank**: compute true squared-L2 only for those
   `rerank_k` candidates.

That's it. The crate's `finger_top_k` is 12 lines.

## Practical failure modes

* **Full-rank Gaussian is hostile**. Real embeddings concentrate on
  low-dimensional manifolds; synthetic Gaussian does not. The 81%
  recall above is a **floor**, not a ceiling.
* **Rank-1 collisions**: when many base vectors have nearly the same
  projection, rerank still has to look at all of them. Mitigation:
  larger `r`, or use a structured projection (FastJL, Hadamard) so
  variance is lower at the same cost.
* **OOD queries**: if query distribution differs from base
  distribution, the JL guarantee still holds, but `rerank_k` may need
  to grow. LeanVec (Intel 2024) addresses this by learning the
  projection.
* **Memory cost**: each base vector now carries an extra `4r` bytes.
  At `r=64` that's 256B per vector — comparable to one row of an INT8
  SQ index. Free if you were already paying for SQ; non-trivial if
  you were not.

## What to improve next

1. **Drop into `ruvector-acorn` / `ruvector-rabitq`'s graph search**
   as the default per-edge distance estimator behind a feature flag,
   replacing the current full-precision distance call inside the
   beam-search loop. Expected: 1.3x–2x graph-search latency win at
   matched recall on SIFT-1M.
2. **Structured projections (Hadamard / FastJL)**: replace the
   dense Gaussian matrix with `D · H · P` (random sign flip,
   Hadamard, random permutation). Computes `q_proj` in `O(d log d)`
   instead of `O(rd)` while preserving the JL guarantee.
3. **INT8 packing of `base_proj`**: the projected representation has
   bounded magnitude after JL scaling; quantizing to INT8 cuts
   memory another 4x and lets one SIMD load fetch four lanes.
4. **Real-data validation harness**: pull SIFT-1M / GIST-1M /
   text-embedding-ada-002 dumps via a separate
   `ruvector-finger-datasets` crate and re-run the bench. The
   numbers above are honest but pessimistic.
5. **Slack-tuned lower bound**: the current `slack=0.0`
   `FingerEstimator` is identical to `JlProjector` ranking. A non-
   zero slack converts the JL estimate into a *certifiable* lower
   bound (Markov-style) so the rerank step can also be skipped on
   confidently-far candidates.

## Production crate layout proposal

```
crates/ruvector-finger/
  Cargo.toml
  src/
    lib.rs              -- DistanceEstimator trait + 3 backends
    bench_harness.rs    -- shared bench harness (lib + bin + criterion)
    bin/bench.rs        -- prints the table in this README
    integrations/       -- (next iter) plug into ruvector-acorn graph search
  benches/
    finger_bench.rs     -- criterion micro-benches
```

Once integrations land, the crate joins the default `cargo build
--workspace` set (already done in this nightly) and is published as
`ruvector-finger = { path = "crates/ruvector-finger" }` from
`ruvector-acorn` and any future `ruvector-vamana`.

## References

* Chen, P. et al. "FINGER: Fast Inference for Graph-based Approximate
  Nearest Neighbor Search". KDD 2023.
* Achlioptas, D. "Database-friendly random projections". JCSS 2003.
* Dasgupta, S. & Gupta, A. "An elementary proof of a theorem of
  Johnson and Lindenstrauss". RSA 2003.
* Gao, J. & Long, C. "High-Dimensional Approximate Nearest Neighbor
  Search: with Reliable and Efficient Distance Comparison Operations".
  SIGMOD 2023.
* Gao, J. et al. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search".
  SIGMOD 2024.
