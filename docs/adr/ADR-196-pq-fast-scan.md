---
adr: 196
title: "PQ FastScan — 4-bit Product Quantization with SIMD shuffle ADC"
status: accepted
date: 2026-05-20
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-195]
tags: [pq, fastscan, ann, vector-search, simd, neon, quantization, recall, nightly-research]
---

# ADR-196 — PQ FastScan: 4-bit Product Quantization with NEON Shuffle ADC

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-20-pq-fast-scan`
as `crates/ruvector-pq-fastscan`. All unit and integration tests pass with
`cargo test --release -p ruvector-pq-fastscan`; build is green with
`cargo build --release -p ruvector-pq-fastscan`.

## Context

The existing PQ implementation in `ruvector-core::advanced_features::product_quantization`
uses 8-bit codes (`K = 256` per sub-quantizer) and a scalar asymmetric-distance
computation (ADC) loop: per database vector, `M` scalar gathers into a `M × 256`
f32 LUT followed by an f32 accumulator. On AArch64 / x86-64 this is bottlenecked
by gather latency, not arithmetic.

The state-of-the-art mitigation has been frozen since 2017 (André, Kermarrec,
Le Scouarnec at ICDE 2017; "Quicker ADC" TPAMI 2021): shrink `K` to 16 so each
sub-quantizer's LUT row fits in a single 16-byte SIMD register, then use one
table-lookup instruction (`vqtbl1q_u8` on AArch64, `pshufb` / `vpshufb` on
x86) to compute 16 distances per cycle. The kernel is reused by ScaNN (Google),
FAISS `IndexPQFastScan` (Meta), Milvus (Knowhere), and Qdrant.

ruvector ships several adjacent ideas — RaBitQ (1-bit), RAIRS-IVF, LVQ — but
the canonical 4-bit FastScan kernel is missing. No crate on crates.io
exposes one stand-alone.

## Decision

Add a new workspace member, `crates/ruvector-pq-fastscan`, with:

1. A reusable `ProductQuantizer` (training and codebook representation;
   identical surface for `K = 256` PQ8 and `K = 16` FastScan).
2. A baseline `Pq8Index` with scalar f32 ADC scan, used as the
   apples-to-apples comparison point.
3. A `FastScanIndex` that stores codes in 32-vector blocks with the
   packed-nibble layout consumed by `vqtbl1q_u8`, with:
   * Scalar reference scan (`scan_block_scalar`) — bit-exact oracle.
   * NEON SIMD scan (`scan_block_neon`) — selected at compile time on
     AArch64. Random-input unit tests assert agreement with the scalar
     oracle.
4. Per-query u8 LUT quantization with per-row minimum subtraction (the
   FAISS/ScaNN trick that prevents outlier collapse — first PoC attempt
   used global scaling and recall@10 was 0.06; fix raised it to 0.86 with
   rerank).
5. Two-stage search: FastScan filter → exact f32 L2 rerank on top-`C`
   candidates. The pattern is standard production use and is the
   recommended public entry point (`FastScanIndex::search_rerank`).

The crate is feature-free and has no async / no-std requirements. NEON is
the only SIMD path implemented; AVX2 / AVX-512 are sketched in the
research doc and deferred to a follow-up.

### Public API

```rust
let pq = ProductQuantizer::train(&train, n_train, dim, m, /*k=*/ 256, iters, seed)?;
let idx = Pq8Index::from_vectors(pq, &data, n);
let top10 = idx.search(query, 10);

let fs = FastScanIndex::from_vectors(&train, n_train, &data, n, dim, m, iters, seed)?;
let lut = fs.build_lut(query);
let top10_raw = fs.search_u16(&lut, 10);
let top10_rer = fs.search_rerank(&lut, query, &data, dim, /*candidates=*/ 100, 10);
```

## Consequences

### Measured

On Apple M4 Max, single-threaded, low-rank Gaussian synthetic (the
SIFT-shape benchmark; see research doc for failure-mode analysis of
isotropic and cluster-mixture alternatives):

| Variant                | n = 100k, d = 128, M = 16, 100 queries | recall@10 |
|------------------------|---------------------------------------:|----------:|
| Flat f32               | 465 ms (215 QPS)                       | 1.000     |
| PQ8 scalar ADC         | 178 ms (562 QPS)                       | 0.600     |
| **FastScan-4 raw**     | **60.5 ms (1,652 QPS)**                | 0.372     |
| **FastScan-4 + rerank-100** | **64.3 ms (1,555 QPS)**           | **0.855** |

Criterion kernel-only (n = 50k, d = 128, M = 16): FastScan **3.1× faster
than scalar PQ8 ADC** (249 µs vs 775 µs); **8.5× faster than flat L2 +
sort** (2.12 ms). Storage: **8 bytes/vec vs 16 bytes/vec for PQ8 and 512
bytes/vec for f32**.

### Positive

* First Rust-native 4-bit FastScan kernel. Closes the parity gap with
  FAISS / ScaNN for the most-cited PQ acceleration of the last decade.
* Codebook representation is shared with the existing PQ8 path —
  downstream consumers (RAIRS, LeanVec) can switch storage layouts
  without re-training.
* The NEON path is 25 lines of `unsafe` confined to one function, with
  a scalar oracle as the regression backstop. Easy to audit.
* Single-threaded baseline; trivially parallelisable across blocks with
  `rayon` (the demo binary stays serial to keep numbers reproducible
  but the `search_u16` block loop is embarrassingly parallel).

### Negative / risks

* **No AVX2 / AVX-512 path.** x86 users get the scalar fallback. This
  is a P0 follow-up.
* **K-means init is random-point.** Adequate for `K = 16`, marginal
  for `K = 256`. Production code should switch to k-means++ before
  shipping a stable API.
* **u8 LUT quantization loses precision in the deep tail.** Raw
  FastScan recall plateaus around 0.40 on this benchmark; the rerank
  stage is essentially mandatory for production recall.
* **One outlier centroid in a single sub-quantizer used to collapse
  recall to ~0.06** before the per-row min subtraction was added.
  Documented; the test suite uses real LUTs so a regression would be
  caught immediately. But the failure mode is silent (no crash, just
  bad recall) and worth keeping in mind for future LUT changes.

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| **Stay on PQ8 scalar** | 3× throughput gap is too large to ignore; same byte budget is achievable at 2× storage cost via FastScan-4. |
| **Wrap FAISS via FFI** | Adds a C++ build dependency to ruvector; contradicts the workspace's no-C++ policy; loses the no-std / wasm story. |
| **1-bit RaBitQ instead** | Already shipped in `ruvector-rabitq`. Solves a different point in the precision/throughput curve (faster scan, much lower recall without rotation tricks). PQ FastScan remains industry standard for medium-recall workloads. |
| **Direct GPU implementation first** | High effort, narrower deployment (Metal/Vulkan only). CPU FastScan unlocks every existing ruvector deployment immediately and is the prerequisite for any honest GPU comparison. |
| **Defer until IVF integration** | The kernel is reusable across flat, IVF, and graph-quantized backends. Landing it stand-alone with a clean unit-test surface is the smallest credible increment. |

## Follow-ups (tracked in research doc)

1. AVX2 + AVX-512 scan paths (P0).
2. Wire FastScan into `ruvector-rairs` posting-list scan (P0).
3. Anisotropic PQ training (ScaNN loss) — P1.
4. OPQ rotation pre-multiplication — P1.
5. k-means++ initialisation — P1.
6. Residual PQ ("PQ4 + PQ4") — P2.
7. Metal/WGSL kernel — P3.
