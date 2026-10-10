# PRVQ: Product-Residual Vector Quantization Cascade Rerank

**Date:** 2026-10-10
**Status:** Prototype, measured
**Crate:** `crates/ruvector-rvq-cascade`
**ADR:** [ADR-362](../../../adr/ADR-362-rvq-cascade-rerank.md)

## Abstract

Classic IVF-ADC pipelines perform PQ-ADC filtering to produce a candidate
shortlist, then full-precision L2 verification. The verification step is
the dominant cost when vectors live on disk/NVMe or when `d` is large
(SIMD doesn't help compressed codes but does nothing for the float fetch
step). This research adds a **Residual Vector Quantization (RVQ)** code
on top of PQ — a `K`-stage 8-bit cascade quantizing the PQ residual —
and derives a composite distance estimator that strictly dominates PQ
ADC. On a 20 000 × 128-dim Gaussian-mixture corpus, PRVQ reaches
recall@10 ≥ 0.978 using **2× fewer full-precision verifies** than the
classic PQ → full-verify rerank.

## SOTA survey (as of 2026)

| Line of work | Reference | Core idea |
|---|---|---|
| Product Quantization | Jégou, Douze, Schmid '11 | Decompose `d` into `m` subspaces, each 8-bit |
| Optimized PQ | Ge et al. '13 | Rotate space before PQ (ADR OPQ-rotation) |
| Additive Quantization | Babenko & Lempitsky '14 | `K` codebooks, sum reconstructs vector |
| Composite Quantization | Zhang et al. '14 | AQ with orthogonality constraint for fast ADC |
| **PRVQ** (ours) | this doc | PQ as backbone + RVQ(residual) refinement |
| RaBitQ | Gao & Long '24 | Random-rotation + binary quantization, PAC bounds |
| ADC-rerank disk | Milvus '25 / Qdrant '25 release notes | Separate RAM-resident code & disk float |

The composite-quantization family is well-explored, but production
engines (Milvus, Qdrant, Weaviate, Pinecone) ship 8-bit PQ + rerank by
default. **PRVQ sits in a cheap niche**: adds `K` bytes/vec, keeps the
PQ decoder, cuts the single most expensive step (full-precision fetch).

## Proposed design

Per vector `x`:
1. PQ encode: `x_pq = ⊕_sub cent_{sub}[code^{PQ}_{sub}(x)]`
2. Residual `r = x − x_pq`
3. RVQ encode the residual with `K` whole-vector 8-bit codebooks:
   `r_rvq = Σ_{s<K} cent^{(s)}[code^{(s)}(r_{s−1})]`, with
   `r_{s} = r_{s−1} − cent^{(s)}[code^{(s)}(r_{s−1})]`, `r_{−1}=r`.

At query time:
- Build PQ L2 LUT (`m × 256 × subdim` ops, once per query).
- Build `K` RVQ dot LUTs (`K × 256 × d` ops).
- Estimate `l2sq(q, x) ≈ ADC_pq(q, x) + ‖r_rvq‖² + 2⟨x_pq, r_rvq⟩
  − 2⟨q, r_rvq⟩` using one `ADC_pq` + per-vector cached
  `‖r_rvq‖² + 2⟨x_pq, r_rvq⟩` + `K` LUT lookups.

Per-candidate query cost: `m + K` lookups + 1 cached-float fetch.
Compared to full-precision verify: `d` multiply-adds + a `d · 4`-byte
fetch.

The composite estimator is strictly more accurate than `ADC_pq` because
RVQ residual terms are trained to minimize reconstruction error, so the
RVQ contribution to the estimate has mean zero on the training
distribution and reduces variance of the final estimate.

## Implementation notes

Single standalone crate, no SIMD, portable scalar Rust so numbers
reproduce bit-exactly. Trait-free; the structs are concrete so a
later ADR can introduce a `DistanceEstimator` trait when SIMD / GPU
backends land.

- Codebook training: k-means++ init + 12 Lloyd iterations, `ChaCha20Rng`
  seeded.
- Residual cross-term `2⟨x_pq, r_rvq⟩` is precomputed per vector so
  the online estimator is O(`m + K`) in work.
- Partial RVQ norms `‖sum_{s≤s'} centroid_s‖²` are also precomputed per
  (vector, stage) so a mid-cascade estimate stays O(1) in `d`.

Code is in `crates/ruvector-rvq-cascade`, 450 lines, under the 500-line
cap.

## Benchmark methodology

- Workload: Gaussian-mixture synthetic data (100 clusters, intra-cluster
  σ small), N=20 000, d=128, nq=200, k=10.
- Hardware: Apple M-series (scalar Rust `release` build, LTO=thin).
- Ground truth: brute-force L2 top-10 per query.
- Metric: recall@10 vs ms/query, plus **full-precision verifies per
  query** (the physical read count that dominates disk-resident ANN).

## Results

```
=== RVQ cascade bench (scalar, N=20000, d=128, nq=200, k=10) ===
Train: PQ-8: 1.819s  PRVQ (m=8, K=4): 11.031s
Memory: PQ = 284 KB  PRVQ = 1265 KB  float32 raw = 10000 KB

Variant                                      ms/query    recall@10
------------------------------------------------------------------
A. Full float32 brute                           0.913        1.000
B. PQ-ADC only (no rerank)                      0.088        0.150
C. PQ -> full-verify top-20 (fetches=20)        0.089        0.241
C. PQ -> full-verify top-50 (fetches=50)        0.092        0.430
C. PQ -> full-verify top-100 (fetches=100)      0.093        0.654
C. PQ -> full-verify top-200 (fetches=200)      0.103        0.913
C. PQ -> full-verify top-400 (fetches=400)      0.115        0.978
C. PQ -> full-verify top-800 (fetches=800)      0.138        0.997
C. PQ -> full-verify top-1600 (fetches=1600)      0.182        0.999
D. PRVQ cascade pq=2000 verify=20               0.129        0.451
D. PRVQ cascade pq=2000 verify=50               0.128        0.715
D. PRVQ cascade pq=2000 verify=100              0.137        0.879
D. PRVQ cascade pq=2000 verify=200              0.143        0.978
D. PRVQ cascade pq=2000 verify=400              0.150        0.994

At recall >= 0.95:
  Classic PQ-rerank: top-400 shortlist -> 400 fetches, 0.115 ms, recall 0.978
  PRVQ cascade: pq=2000 verify=200 -> 0.143 ms, recall 0.978
  Fetch reduction (PRVQ vs classic at same recall): 2.0x fewer full fetches
```

**Headline:** at recall@10 = 0.978 PRVQ halves the full-precision
fetches (200 vs 400). In-RAM latency pays a small penalty (0.143ms vs
0.115ms) because the RVQ dot-LUT build and composite estimator add
work; **in a disk-resident DiskANN setting each full fetch is a page
I/O (~15 µs NVMe, ~1 ms rotational), so halving fetches shaves 3 ms
per query at the typical disk tier.**

## How it works (walkthrough)

1. **Train PQ** on raw vectors as usual. Each vector is now 8 bytes.
2. **Compute PQ residuals**: for every vector, subtract the PQ
   reconstruction. The residual carries the "error" PQ couldn't fit.
3. **Train RVQ on the residuals**: stage 0 is a plain 256-centroid
   k-means on the residual vectors. Stage 1 runs k-means on the
   stage-0 residual-of-residual. K stages produce a sum
   `cent₀ + cent₁ + … + cent_{K−1}` that reconstructs the PQ residual
   with progressively lower error.
4. **Cross-term bookkeeping**: at build time we also store
   `2 · ⟨x_pq, r_rvq⟩` and the full RVQ-reconstruction norm per vector,
   so the online estimator never touches `d` floats per candidate — it
   just adds a cached scalar plus `K` LUT lookups.
5. **Query**: compute the PQ L2 LUT and K RVQ dot LUTs. For each
   candidate, compute `ADC_pq + ‖r_rvq‖² + 2⟨x_pq, r_rvq⟩ − 2⟨q, r_rvq⟩`.
   This is `m + K + 2` scalar ops per candidate (vs `d` for full
   verify). Keep the top-N by this estimate; full-verify just those N.

The hidden-but-important detail: RVQ residual terms have **mean-zero
dot product with the query** when the codebooks are well-trained, so
the estimator reduces variance without introducing bias. The partial-
norm bookkeeping lets early-cascade estimates stay cheap, which is why
large PQ shortlists (pq=2000) can be refined faster than full-verify
would.

## Practical failure modes

- **Short vectors (`d ≤ 32`)**: PQ already saturates — PRVQ adds
  overhead with no accuracy headroom. Fall back to plain PQ-rerank.
- **Highly skewed clusters**: RVQ k-means on residuals converges
  slowly; may need > 12 Lloyd iterations.
- **High-K regimes (`K ≥ 16`)**: diminishing returns per stage and
  training time balloons. The sweet spot is `K ∈ {2, 4, 8}`.
- **Query norms drifting from training distribution**: the mean-zero
  residual-dot assumption weakens; estimator bias grows. Mitigate with
  per-cell training (one PRVQ per IVF cell).

## What to improve next

1. **Per-cell PRVQ** inside IVF to tighten the mean-zero assumption.
2. **SIMD AVX2/NEON** LUT scan (16-way packed) — present code is scalar
   deliberately for reproducibility; the hot loop is a `K`-way LUT
   stream begging for FastScan.
3. **Opt-in float16 / int8 for the cached cross-term** to halve
   per-vector overhead.
4. **Fused OPQ rotation** before PQ to reduce stage-0 PQ error, which
   amplifies PRVQ's gain.
5. **Learned cascade thresholds** so verify count adapts to the query
   difficulty (hook into `ruvector-dyn-rerank-budget`).

## Production crate layout

If promoted out of nightly research, the natural integration is:

```
crates/ruvector-prvq/
  src/
    lib.rs           // Pq8, RvqCascade, Prvq, CascadePipeline
    simd/            // feature-gated AVX2 / NEON LUT scan
    train.rs         // k-means variants (Lloyd / mini-batch / HGM)
  benches/
    bench.rs
  examples/
    integrate_ivf.rs // plug into ruvector-core::ivf for E2E recall
```

The core IVF path keeps `Pq8`; adding `Prvq` is purely additive. A
feature flag `prvq` turns it on at index-build time; the on-disk layout
appends `K` bytes + 8 scalar bytes per vector.

## References

- Jégou, Douze, Schmid. "Product quantization for nearest neighbor
  search." PAMI 2011.
- Ge et al. "Optimized Product Quantization." PAMI 2013.
- Babenko, Lempitsky. "Additive Quantization for Extreme Vector
  Compression." CVPR 2014.
- Zhang et al. "Composite Quantization for Approximate Nearest Neighbor
  Search." ICML 2014.
- Gao, Long. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  SIGMOD 2024.
- Jayaram Subramanya et al. "DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node." NeurIPS 2019.
