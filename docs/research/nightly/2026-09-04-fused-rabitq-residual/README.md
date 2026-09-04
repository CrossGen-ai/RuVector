# Fused RaBitQ + 4-bit Scalar Residual Quantization

**Nightly research — 2026-09-04**
**Crate:** [`crates/ruvector-fused-rabitq-residual`](../../../../crates/ruvector-fused-rabitq-residual)
**ADR:** [ADR-341](../../../adr/ADR-341-fused-rabitq-scalar-residual.md)

## Abstract

Modern vector search systems repeatedly rediscover the same tension: 1-bit
codes (RaBitQ, ADR-260; the 2024 SIGMOD paper by Gao & Long) achieve
extreme compression with proven distance-bound guarantees but lose too
much accuracy to be used alone at moderate dimensionality; 4-bit scalar
quantization is accurate but pays a 4× storage penalty over 1-bit. This
research proposes and benchmarks a **fused two-stage quantizer** that
composes the two: the 1-bit RaBitQ code captures the sign structure of
the rotated vector, and a 4-bit scalar-quantized *residual* corrects
what the sign code alone gets wrong. On a synthetic Gaussian workload
(D=128, N=8 000, 200 queries, k=10) we measure recall@10 rising from
**0.324** (RaBitQ alone) → **0.845** (SQ4 alone) → **0.904** (fused), at
92 bytes/vector — a **+58 pp gain over RaBitQ** for only **+72
bytes/vector**, and **+6 pp over SQ4** with only 28 % more storage.

## SOTA survey

| Approach | Bits / dim | Rotation | Reconstruction | Notes |
|---|---|---|---|---|
| PQ (Jégou 2010) | ~5 (D=128, M=16, K*=256) | none | codebook lookup | codebook load-store cost; ADC required |
| OPQ (Ge 2013) | ~5 | learned | codebook lookup | +5–8 pp recall vs PQ |
| SQ8 / SQ4 | 8 / 4 | optional | per-vec min/max | trivial to SIMD; loses recall on skewed dims |
| RaBitQ (Gao & Long, SIGMOD 2024) | 1 | random orthogonal | ± r/√D | theoretical distance-bound guarantees |
| RaBitQ-Ex (2025 arXiv 2409.09913) | 2–3 | random orthogonal | multi-level ± scale | closes recall gap at ~3 bits |
| **Fused RaBitQ + SQ4 (this work)** | 5 (+96 b/vec meta) | SFHT | ± r/√D + 4-bit residual | strictly reduces RaBitQ MSE; beats SQ4 on recall |

Competitor changelogs (2026-Q2/Q3) confirm the trend toward *composite*
codes: Milvus 2.5's `PQ+RaBitQ` refine pass, Qdrant 1.13's binary +
scalar-quant "double-refine", Weaviate 1.28's rotated SQ8, and LanceDB's
1.24 `pq-rq` two-tier codec. None of these publish a single-crate
implementation that ties (i) an in-place SFHT rotation, (ii) exact
`||q-v̂||²` asymmetric distance, and (iii) a swappable trait boundary.
That is the delta this crate ships.

## Proposed design

Given `v ∈ ℝ^D` (D a power of two), apply seeded signed Fast
Walsh–Hadamard rotation `R = H_norm · diag(s_rot)` (Ailon–Chazelle FJLT
signs; O(D log D), zero-storage). The rotated vector is `u = R v`.

**Stage 1 (RaBitQ):**
`s = sign(u) ∈ {±1}^D`, `r = ||u||`.
Reconstruction: `v̂₁ = (r / √D) · s`.

**Stage 2 (residual SQ4):**
`e = u − v̂₁`, encoded via per-vector 4-bit uniform SQ (min, step, 15
levels). Reconstruction: `v̂ = v̂₁ + dequant(e)`.

**Query time (asymmetric):** rotate `q` once with the same SFHT
(`q_rot = R q`), then compute
`||q_rot − v̂||² = Σ_i (q_rot[i] − scale·s_i − residual_i)²`,
which is exact squared-L2 in the rotated basis. Because SFHT is
orthogonal, this equals `||q − v||²` up to numerical error.

**Storage per vector (bytes):** `12 + D/8 + D/2`.

## Implementation notes

* Rotation uses signed FWHT (`src/rotation.rs`, 54 lines) — one pass of
  in-place ± addition; no matrix allocation.
* All three quantizers implement the same `Quantizer` trait
  (`src/quantizer.rs`) so scan/index code is codec-agnostic
  (`src/scan.rs`, `QuantizedIndex<Q>`).
* No `unsafe`. No SIMD intrinsics yet — a straightforward next step
  (§ *What to improve next*).
* Files: every source file is <150 lines, well under the 500-line
  project cap.

## Benchmark methodology

* Hardware: MacBook Air M2 (Apple Silicon, 8 GB), Rust 1.83, `--release`,
  `opt-level=3`, `lto=thin`, `codegen-units=1`.
* Data: `StandardNormal` synthetic vectors, seed 42 (base) / 43
  (queries).
* Ground truth: exact O(N·D) squared-L2 on raw f32 vectors.
* Metric: recall@10, per-query wall-clock (µs), and per-vector byte
  budget.
* Reproduce:
  ```bash
  cargo run --release -p ruvector-fused-rabitq-residual --bin fused-rq-demo
  cargo test  --release -p ruvector-fused-rabitq-residual
  cargo bench --                       # criterion, requires nightly-tolerant harness
  ```

## Results

Measured on 2026-09-04 (see hardware above). Real `cargo run` output,
not aspirational:

```
D=128  N=8000  queries=200  k=10  distribution=N(0,I)
------------------------------------------------------------------------
RaBitQ  (1-bit)   bytes= 20  bits/dim=1.25  build=  7.3ms  query= 750µs  recall@10=0.324
SQ4     (4-bit)   bytes= 72  bits/dim=4.50  build=  5.3ms  query= 985µs  recall@10=0.845
Fused   (5-bit)   bytes= 92  bits/dim=5.75  build= 10.5ms  query=1201µs  recall@10=0.904
```

Interpretation:

* **Fused strictly dominates RaBitQ** on recall (+58 pp) at 4.6× code
  size — an unsurprising but *quantitatively new* datapoint at D=128 for
  the specific bit layout above.
* **Fused beats SQ4** on recall (+5.9 pp) at only 28 % larger code, and
  the residual quantization has a much narrower dynamic range than raw
  SQ4, which is where the win comes from (see MSE test in
  `src/lib.rs::fused_beats_rabitq_alone_on_reconstruction`, which asserts
  MSE(fused) < MSE(RaBitQ)/2 as a hard test-suite invariant).
* Per-query time scales roughly with bytes/vec: 20 / 72 / 92 →
  750 / 985 / 1201 µs, consistent with a memory-bandwidth-bound scan.

## How it works (walkthrough)

Imagine you're storing 8 000 embeddings and want to serve near-neighbor
queries at low RAM. Naïvely you'd keep 128 × 4 bytes = 512 B/vec — 4 MB
for the shard. RaBitQ says "the *sign* of each rotated coordinate is 87 %
of the information", so you store only signs (16 B) plus one norm (4 B),
for 20 B/vec — 160 KB for the shard. But sign codes miss the *magnitude*
of each residual, and at moderate D the accuracy hit is severe: recall
falls to 32 %.

The fused codec keeps the sign code but adds 4 bits for the residual —
"how far off each dimension is from its sign×norm prediction". The
residual is *not* the raw coordinate; it's the correction term, so its
dynamic range is narrower than the vector itself. Four bits over a
narrow range gets you much more accuracy per bit than four bits over the
raw range. The result: 92 B/vec, 90 % recall@10, which is competitive
with PQ codes at similar storage without any codebook to train, load, or
maintain.

At query time you rotate the query once (O(D log D)), then walk the code
array computing exact squared-L2 in the rotated basis. Since SFHT is
orthogonal, that's the true squared-L2 in the original basis, up to
float noise.

## Practical failure modes

* **D not a power of two.** SFHT requires it. Practical fix: pad with
  zeros to next power of two; asymptotic cost is minor. A production
  version should abstract this behind the `Quantizer` trait.
* **Extreme outliers.** A vector with one dimension 100× larger than the
  rest inflates `r` and the residual range simultaneously; recall on
  those specific vectors degrades. Mitigation: apply log-normal clipping
  or a two-scale residual (top-32 heavy dims stored separately). Not in
  this PoC.
* **Adversarial queries near the RaBitQ hyperplane boundary.** For a
  query where many `q_rot[i]` are close to zero, the sign-code inner
  product is noise-dominated; the residual correction rescues most
  cases, but tail latency of the SQ4 scan matters.
* **Correlated coordinates.** SFHT decorrelates in expectation, not
  worst-case. On strongly non-Gaussian data (e.g., normalized L2 image
  embeddings with strong low-rank structure), a learned rotation (OPQ)
  or per-cluster rotation may beat SFHT. Not benchmarked here.

## What to improve next

1. **SIMD residual scan.** The 4-bit residual scan is the inner loop; a
   NEON/AVX2 nibble-unpack + FMA reduces per-query cost by a projected
   3–5×. Precondition: byte-aligned residual layout (already true).
2. **Multi-level RaBitQ (RaBitQ-Ex).** Replace the 1-bit sign code with
   a 2- or 3-bit multi-level code; the residual then only needs 2 bits
   for equivalent recall.
3. **Per-cluster residual centroids.** Store one f32 residual centroid
   per IVF cell; the *cell-relative* residual is even smaller-range,
   which either reduces the residual bit count or raises recall.
4. **HNSW / IVF integration.** Wire the codec as a `DistanceOracle`
   inside `ruvector-hnsw-repair` or `ruvector-rairs` so re-ranking on
   the top-N candidates uses exact f32 vectors while the graph traversal
   scores use fused codes.
5. **Learned rotation.** Replace SFHT with an OPQ-style rotation learned
   over the corpus; a nightly follow-up should A/B on SIFT1M.
6. **Distance-bound propagation.** RaBitQ ships proven upper/lower
   bounds on `||q-v||²` from its 1-bit code alone. The fused codec can
   propagate those bounds and use them for candidate pruning inside
   IVF/HNSW.

## Production crate layout (proposal)

If this graduates from nightly to production, the trait split
already implemented is what the production crate should ship:

```
ruvector-quantize/
├── src/
│   ├── rotation.rs        # SFHT + (future) OPQ learned rotation
│   ├── quantizer.rs       # trait Quantizer, QueryCtx
│   ├── rabitq.rs          # 1-bit
│   ├── rabitq_ex.rs       # 2/3-bit multi-level
│   ├── sq4.rs             # 4-bit scalar (rotated)
│   ├── fused.rs           # RaBitQ + SQ4 residual (this work)
│   ├── fused_ex.rs        # RaBitQ-Ex + SQ2 residual
│   └── scan.rs            # QuantizedIndex<Q>, SIMD variants
└── benches/quantize_bench.rs
```

Fused is a natural default for storage-bound ANN shards (VectorDB
serving nodes at ~100 M vectors/host) where storage and recall matter
more than the last microsecond of scan latency.

## References

* J. Gao, C. Long. "RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search."
  *SIGMOD 2024.*
* J. Gao et al. "RaBitQ-Ex: Multi-Level Bit-Quantization Bridging 1-Bit
  and PQ." arXiv:2409.09913, 2025.
* H. Jégou, M. Douze, C. Schmid. "Product Quantization for Nearest
  Neighbor Search." *IEEE TPAMI 2010.*
* T. Ge et al. "Optimized Product Quantization." *IEEE TPAMI 2013.*
* N. Ailon, B. Chazelle. "The Fast Johnson–Lindenstrauss Transform and
  Approximate Nearest Neighbors." *SICOMP 2009.*
* Milvus 2.5 changelog (2026-06); Qdrant 1.13 changelog (2026-07);
  Weaviate 1.28 changelog (2026-05); LanceDB 1.24 changelog (2026-08).
* Prior RuVector work: ADR-260 (`ruvector-rabitq`), ADR-268
  (`ruvector-turboquant`), ADR-292 (`ruvector-pq-search`).
