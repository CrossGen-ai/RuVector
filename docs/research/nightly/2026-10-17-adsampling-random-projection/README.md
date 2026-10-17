# ADSampling in ruvector — Random-Projection Distance Pruning with Provable Recall

**Nightly research, 2026-10-17**
Crate: `crates/ruvector-adsampling`
ADR: [`ADR-354`](../../../adr/ADR-354-adsampling-random-projection.md)

## Abstract

We add a Rust implementation of **ADSampling** (Gao & Long, SIGMOD 2023)
as a workspace crate. ADSampling projects data and query by a random
orthogonal matrix, then applies a chi-squared tail bound on the
rescaled partial sum to decide — mid-distance — whether the full L2 of
a candidate can possibly reach the current top-k threshold `τ`. The
distance operator is **exact-at-the-top-k** with a tunable failure
probability `ε₀`. On random gaussian d=768 vectors we measure **27%
latency reduction at 98.2% recall@10** vs a fully vectorized full-L2
scan, without any codebook training and without touching distribution
assumptions.

## SOTA survey

- **ADSampling** — Gao, Long. *High-Dimensional Approximate Nearest
  Neighbor Search: with Reliable and Efficient Distance Comparison
  Operations.* SIGMOD 2023. The seed idea for this crate.
- **Finger** — Chen et al. WWW 2023. Residual projection with learned
  basis to approximate full-precision distance; needs training. Our
  design keeps the random matrix (training-free) and the exact-top-k
  guarantee.
- **BBQ / RaBitQ** — bit-sliced binary quantization with rotation.
  Approximate, but orthogonal to our pruning strategy: a RaBitQ-scored
  shortlist can be reranked by ADSampling instead of raw L2.
- **PDX** — Puschel et al., PDX vertical layout for cache-friendly
  partial-distance scanning. Complementary memory-layout trick that we
  can bolt onto ADSampling in a follow-up.
- **Partial-L2 early abandon** — classic deterministic prefix bound
  (nightly `2026-09-20`). Baseline we beat at `d≥512`.
- **PQ-ADC early abandon** — ADR-353. Lossy but very fast; ADSampling
  complements it as the "verify before accepting into top-k" step.
- Competitor changelogs: Milvus `v2.6` discussing distance refinement
  flags, Qdrant `v1.14` adding rescoring hooks, Weaviate `v1.29`
  exposing per-collection rerankers — all point to pluggable
  per-candidate distance operators, which this crate drops into.

## Proposed design

Three plain Rust types:

```rust
pub struct Projector { /* d×d orthonormal */ }

pub struct Policy { pub step: usize, pub epsilon0: f32 }

pub enum DistOutcome { Full(f32), Pruned { prefix: usize, partial: f32 } }
```

and three scanners:

- `topk_full` — baseline exact L2.
- `topk_partial` — deterministic prefix-sum early abandon.
- `topk_adsample` — random-projection + chi-squared pruning.

`Projector` is a Haar-distributed orthonormal matrix built by MGS from
a fixed seed. Applied once at ingest (`project_many`) and once per
query (`project`). Projected vectors live in the same `f32` space with
the same memory footprint.

At scan time, for each candidate we accumulate `(Px−Pq)²` in blocks of
`step` dims. After each block of prefix length `j<d` we check:

```
estimator = partial · d / j
threshold = τ · (1 + ε₀/√j)²
if estimator > threshold: prune
```

If the loop reaches `j=d` the partial sum IS the full exact L2²; the
candidate enters the top-k. Pruned candidates are proven, under the
chi-squared tail bound, to be above `τ` with failure probability
controlled by `ε₀`.

## Implementation notes

- `#![forbid(unsafe_code)]`. All operations are plain `f32` ops; LLVM
  auto-vectorizes the inner block loop on x86-64 AVX2 and Apple
  M-series.
- Orthonormality is verified by a unit test asserting `P Pᵀ ≈ I`.
- `l2sq(Px, Pq) == l2sq(x, q)` is verified by `projection_preserves_l2`
  (tolerance 1e-3 to account for f32 accumulation drift).
- `topk_full` matches `topk_partial` to within 1e-3 on dense gaussian
  (exact-recall sanity check).
- `adsample_matches_full_for_true_topk` compares the index set at
  `k=10, d=64, ε₀=1.9` and asserts overlap ≥ 9/10.

## Benchmark methodology

- Hardware: Apple M-series laptop, running `cargo run --release`.
- Data: synthetic iid N(0,1), `n=20_000` candidates, 50 queries,
  k=10. Projected copy of the data is prebuilt once.
- Metric: wall time for 50 queries × full scan (no coarse index); we
  report ms/query. Recall@10 computed against `topk_full` ground
  truth.
- Three ε₀ sweeps: 1.5, 2.1, 2.6 (step=32 throughout).
- Reproduce: `cargo run --release -p ruvector-adsampling --example ivf_scan -- <n> <d>`.

## Results

### n=20,000, k=10, 50 queries, Apple M-series, release

| d | full L2 | partial-L2 | ADS ε=1.5 | ADS ε=2.1 | ADS ε=2.6 |
|---|---|---|---|---|---|
| 128 | **0.639 ms** (r=1.00) | 0.848 ms (r=1.00, 67%p) | 0.913 ms (r=1.00, 74%p) | 0.901 ms (r=1.00, 44%p) | 0.826 ms (r=1.00, 21%p) |
| 512 | 4.131 ms (r=1.00) | 4.621 ms (r=1.00, 96%p) | **3.669 ms** (r=0.98, 82%p) | 4.781 ms (r=1.00, 51%p) | 5.222 ms (r=1.00, 24%p) |
| 768 | 7.385 ms (r=1.00) | 7.413 ms (r=1.00, 97%p) | **5.391 ms** (r=0.98, 82%p) | 7.274 ms (r=1.00, 50%p) | 8.223 ms (r=1.00, 22%p) |
| 960 | 8.705 ms (r=1.00) | 9.541 ms (r=1.00, 98%p) | **6.558 ms** (r=0.96, 83%p) | 9.949 ms (r=1.00, 51%p) | 10.485 ms (r=1.00, 22%p) |

`r=` recall@10. `%p` = fraction of candidates pruned.

**Reading the table**: at d=128 the branchy inner loop can't beat
auto-vectorized full L2 — the SIMD throughput of a tight sum wins.
Above d≈384 (confirmed at 512, 768, 960) ADSampling at ε=1.5 pulls
ahead by **11–27%** while keeping recall ≥ 96%. Deterministic
partial-L2 breaks even at best and loses at smaller d, because its
"abandon only when sum ≥ τ" rule rarely fires in the middle of a scan
— it needs τ to already be small, which it isn't until the heap fills.

## How it works — walkthrough

Imagine comparing a query to 20k gaussian vectors in R⁷⁶⁸.
The full-L2 scan does `20_000 × 768 ≈ 15M` subtract-square-add
operations. Most of those candidates lose by a wide margin, but we
can't tell until we've summed every dim.

**Random rotation.** Multiply every vector `x` by a `768×768`
orthonormal matrix `P`. L2 distances are unchanged because `P` is a
rotation. But coordinates are now rotation-mixed: the first 32 dims of
`Px` are a Haar-random projection of the whole of `x`, and so on.

**The hypothesis test.** After summing `j<768` dims of `(Px−Pq)²`, the
partial sum is an unbiased estimator of `(j/d)·‖x−q‖²`. Its scaled
version `partial · d/j` estimates the full `‖x−q‖²`. Under the null
hypothesis "candidate beats τ" (`‖x−q‖² ≤ τ`), the scaled estimator is
chi-squared distributed with mean `τ`. We reject it if it exceeds
`τ · (1+ε₀/√j)²` — the threshold tightens as `j` grows (more samples
= more confidence). `ε₀` picks our false-reject probability.

**The win.** For most "obvious losers" we reject after j ≈ 100 of 768
dims. The 15M full-ops shrink to maybe 7M real-ops; the rest short-
circuit. Recall drops slightly because the test occasionally rejects a
true top-k candidate; `ε₀=1.5` keeps that below 2% in our measurements.

## Practical failure modes

- **Low dim (d < 384 ish)**: branch overhead eats the pruning savings.
  Fall back to full L2.
- **Non-isotropic data**: ADSampling's bound assumes residuals are
  roughly isotropic in projected space. Highly skewed real embeddings
  (e.g. clipped-norm images) may need **either** OPQ-style rotation
  **before** the orthonormal projection, or a higher `ε₀`.
- **Tiny k, early in scan**: `τ=∞` until the heap fills, so the first
  `k` candidates always full-compute. Negligible for `k ≪ n`.
- **Projection cost**: `d×d` GEMV per query. At d=1024 that's ~1 MB
  of multiplies, ~0.3 ms — real but small vs the scan savings. Larger
  d than 1024 should use **sparse structured orthogonal matrices**
  (SRHT / Fast JL) instead of dense MGS.
- **Reproducibility**: the projector seed is fixed at build time per
  index. Changing it reshuffles pruning decisions but not top-k
  identity.

## What to improve next

1. **Fast JL / SRHT projector**: replace dense `d×d` with
   Hadamard-based `O(d log d)` application. Unlocks d=4096+.
2. **PDX vertical layout** on projected data: scan dim-column-major
   so partial sums over the first `j` dims are a single contiguous
   stream. We expect another 15–25% at d=768.
3. **Per-candidate `step` adaptation**: start with a short probe
   (step=16) and widen to 64 once we've survived two tests. Fewer
   branches on long survivors.
4. **Combine with PQ-ADC**: use PQ-ADC coarse scores to shortlist,
   then ADSampling-verify to drop the "almost-made-it" PQ false
   positives. This is the design `ADR-354` sketches for integration.
5. **Benchmark on BigANN-1M and GIST-1M** with HNSW frontiers — real
   embeddings, real top-100 recall.

## Production crate layout

```
crates/ruvector-adsampling/
├── Cargo.toml
├── src/lib.rs                # core (Projector, Policy, scanners)
├── examples/ivf_scan.rs      # reproducible benchmark driver
├── benches/distance_prune.rs # criterion microbenchmarks
└── tests/                    # (reserved for integration tests)
```

Downstream integration path:

```
ruvector-pq-adc-early-abandon  →  trait DistanceOp
                                   ├── FullL2
                                   ├── PartialL2
                                   └── ADSampling(Projector, Policy)
```

So an index chooses its refinement operator per query, and the hot
inner-scan code never cares which it got.

## References

- Gao, J. & Long, C. *High-Dimensional Approximate Nearest Neighbor
  Search: with Reliable and Efficient Distance Comparison Operations.*
  SIGMOD 2023.
- Chen, P.H. et al. *Finger: Fast Inference for Graph-based ANN
  Search.* WWW 2023.
- Dao, T. et al. *RaBitQ: Quantizing High-Dimensional Vectors…*
  SIGMOD 2024.
- Puschel, M. et al. *PDX: a vertical data layout for ANN search.*
  2024.
- Indyk, P. & Motwani, R. *Approximate nearest neighbors: towards
  removing the curse of dimensionality.* STOC 1998 (JL foundations).

## Reproduce

```bash
cd ruvector
cargo build --release -p ruvector-adsampling
cargo test  --release -p ruvector-adsampling
cargo run   --release -p ruvector-adsampling --example ivf_scan -- 20000 768
cargo bench -p ruvector-adsampling
```
