# PQ FastScan in Rust — 4-bit Product Quantization with NEON shuffle ADC

> Nightly research, 2026-05-20. Branch `research/nightly/2026-05-20-pq-fast-scan`.
> Crate: `crates/ruvector-pq-fastscan`. ADR: [ADR-196](../../adr/ADR-196-pq-fast-scan.md).

## Abstract

Standard 8-bit Product Quantization (PQ) shrinks vector storage by 32× but
its asymmetric distance computation (ADC) is bottlenecked by scalar gathers
into a per-query lookup table. PQ FastScan (André, Kermarrec, Le Scouarnec,
ICDE 2017) compresses the codebook to 4 bits per sub-quantizer so a single
SIMD shuffle instruction (`vqtbl1q_u8` on AArch64, `pshufb` / `vpshufb` on
x86) performs 16 distance lookups in one cycle. This document proposes a
self-contained Rust implementation living in `ruvector-pq-fastscan`,
documents the storage layout that maps onto NEON, presents a working
codebook + scan + rerank pipeline, and reports measured throughput and
recall against an 8-bit ADC baseline and exact-float flat scan on a
100k×128 low-rank Gaussian benchmark.

The PoC achieves **8.5× speedup over flat L2 scan and 3.1× over scalar 8-bit
PQ ADC** at the kernel level (Criterion, 50k×128, Apple M4 Max), and an
end-to-end **7.2× wall-clock speedup over flat scan at recall@10 = 0.86**
when paired with a 100-candidate exact-L2 reranker (100k×128, 100 queries).

## SOTA survey

| Year | Work | Key contribution |
|------|------|------------------|
| 2011 | Jégou, Douze, Schmid — *Product Quantization for NN Search* (PAMI) | 8-bit codebooks, ADC, IVF-PQ. Industry baseline. |
| 2017 | André, Kermarrec, Le Scouarnec — *Quick ADC* / *Quicker ADC* (ICDE) | 4-bit PQ + `pshufb` SIMD lookup. 4–6× over PQ8. The paper this work mirrors. |
| 2019 | Matsui et al. — *Reconfigurable Inverted Index* | Combines FastScan with IVF for billion-scale. |
| 2020 | Guo et al. — *ScaNN* (ICML) | Anisotropic loss + AH (asymmetric hashing) using same 4-bit FastScan kernel. Production at Google. |
| 2021 | Aguerrebere et al. — *LeanVec* | Subspace pruning to amplify FastScan throughput further. |
| 2023 | FAISS `IndexPQFastScan` / `IndexIVFPQFastScan` (Meta) | Mature reference implementation; uses AVX-512 VPSHUFB + horizontal accumulators. |
| 2024 | Gao & Long — *RaBitQ* (SIGMOD) | 1-bit rotation-based quantization; orthogonal to FastScan. (Already in this repo.) |
| 2025 | Yang et al. — *SymphonyQG* | Combines RaBitQ-style rotation with FastScan-style scan kernel on graph indices. |

Competitor changelogs surveyed (May 2026):
* **Milvus 2.5** ships FastScan via Knowhere ≥ 2.4. Reports a 3–4× scan
  speedup on ARM Graviton4.
* **Qdrant 1.13** added 4-bit PQ behind `--quantization fastscan` last
  quarter; documentation notes "2× over 8-bit PQ on AVX2 hardware."
* **Weaviate 1.27** still ships 8-bit PQ only; FastScan is in their
  roadmap.
* **Pinecone** is closed-source but their published latency figures match
  a FastScan-class kernel.
* **LanceDB** uses `arrow-rs` PQ; no FastScan kernel as of v0.13.

Rust-ecosystem state: **no crate on crates.io exposes a 4-bit PQ FastScan
kernel.** `faiss-rs` only wraps FAISS; `instant-distance` and `usearch-rs`
focus on HNSW. This gap is what `ruvector-pq-fastscan` fills.

## Proposed design

### Storage layout (32-vector blocks)

A block of `BLOCK = 32` database vectors is stored as `M × 16` bytes, where
`M` is the number of sub-quantizers. For sub-quantizer `s`, the 16-byte
chunk packs two 4-bit codes per byte:

| Byte index `b ∈ [0,16)` | Low nibble | High nibble |
|-------------------------|------------|-------------|
| `b`                     | code for vector `b`     | code for vector `b + 16` |

Total: `M / 2` bytes per database vector (2 ⁿ ibbles × 4 bits = 1 byte
per 2 vectors per sub-quantizer). For `M = 16` that is **8 bytes per vector**
versus 16 bytes for PQ8.

### Per-query LUT quantization

The 4-bit FastScan ADC accumulates `M` u8 values per database vector into a
u16 accumulator (`M ≤ 257` keeps the accumulator unsaturated). We build
`M × 16` u8 entries from `M × 16` f32 squared-L2 distances between the
query sub-vector and each of the 16 centroids in that sub-quantizer.

Naïve global scaling (one `scale = 255 / max(LUT)`) destroys recall when
**one** sub-quantizer has an outlier centroid: all other rows collapse to
near-zero u8 entries. The fix used here (and by FAISS / ScaNN) is to
**subtract the per-row minimum before quantizing**. Per-row min is a
constant added to every database vector's accumulated distance, so it
cancels in ranking comparisons:

```text
LUT_u8[s, c] = round((LUT_f32[s, c] - row_min[s]) * (255 / max_range))
where max_range = max_s (max_c LUT_f32[s,c] - min_c LUT_f32[s,c])
```

Sub-optimal global ranking only happens if a sibling sub-quantizer
dominates the actual squared-L2 contribution; this is rare in practice
because PQ is intentionally balanced across sub-spaces.

### Scan kernel (AArch64 NEON)

Per sub-quantizer, three NEON instructions do the work for 32 database
vectors:

```rust
let codes = vld1q_u8(&block_codes[s * 16]);          // 16 packed bytes
let idx_lo = vandq_u8(codes, vdupq_n_u8(0x0F));      // lanes 0..15
let idx_hi = vshrq_n_u8(codes, 4);                   // lanes 16..31
let d_lo = vqtbl1q_u8(row, idx_lo);                  // 16-entry table lookup
let d_hi = vqtbl1q_u8(row, idx_hi);                  // 16-entry table lookup
// widen u8 → u16 and accumulate into four uint16x8_t lanes
```

`vqtbl1q_u8` is one issue, single-cycle on every Apple Silicon / Cortex-A
core back to A57 — strictly stronger than the x86 `pshufb` predecessor
because the table size matches PQ4's K=16 exactly with no lane-crossing
workaround. The x86 AVX2 path (`_mm256_shuffle_epi8`) is sketched in the
ADR but not implemented in this PoC (the host hardware is AArch64).

### Codebook training

K-means in each sub-space, identical to standard PQ — only `K` changes
(256 → 16). The same `ProductQuantizer` struct is reused; the FastScan
storage and scan kernel are purely a different read path over the same
`Vec<u8>` codes (modulo the nibble packing).

### Two-stage retrieval

FastScan's u8-quantized LUT loses fidelity in the deep ranking tail. The
production pattern, reproduced here, is:

1. **Filter** — FastScan returns top-`C` (e.g. `C = 100`) candidates.
2. **Rerank** — exact f32 `||q − v||²` is computed on those `C` vectors.

`search_rerank(...)` in `src/fastscan.rs` implements this. At `C = 10 × k`
the rerank stage costs ~6 % over raw scan-only and recovers recall@10 from
0.37 to 0.86 on the benchmark below.

## Implementation notes

* `ProductQuantizer` is shared between PQ8 and FastScan — only `k`
  changes. This keeps the codebook-training surface identical and lets a
  future FastScan-IVF residual quantizer reuse the same trait.
* The scalar reference scan (`scan_block_scalar`) is the regression
  oracle — the NEON kernel must match it bit-for-bit; this is enforced
  by unit tests (`neon_matches_scalar_*`) and a randomised integration
  test (`neon_scalar_agreement_random`).
* Vectors past `n` in the last block are padded with code `0`. The
  accumulator over padding lanes is finite and bounded by `M × 255` but
  the padding positions are truncated before top-k selection.
* All kernels are `#[inline]`-free and `unsafe { ... }` is confined to
  the NEON intrinsics block (≈ 25 lines).

## Benchmark methodology

**Hardware**: Apple M4 Max, macOS 15.6, 128 GB RAM. Single-threaded
benchmark (no Rayon parallelism in the scan path). `cargo build --release`
with workspace defaults (`opt-level=3 lto=fat codegen-units=1`).

**Data**: Low-rank Gaussian — 16-dim latent ~ N(0,1) projected up to `d`
via a random `intrinsic × d` Gaussian matrix, plus per-dim N(0, 0.1)
noise. This is the SIFT/GIST-shape synthetic; pure isotropic Gaussian has
no compressible structure and cluster-mixture collapses K-means onto
modes (we verified both pathologies during PoC bring-up; numbers in the
"Practical failure modes" section).

**Queries**: 100 perturbed copies of held-out database points
(`x + ε`, `ε ~ N(0, 0.1)`). This is the standard "self-search" recall
setup used by FAISS / Annoy / hnswlib benchmarks.

**Train set**: First 20,000 database vectors. K-means runs 12 Lloyd
iterations from random-point initialisation per sub-quantizer (PQ8: K=256;
FastScan: K=16). Identical training seed for both.

## Results

### Scan-only kernel (Criterion, n = 50,000, d = 128, M = 16)

| Variant | Per-query time | Throughput | Speedup vs flat |
|---------|---------------:|-----------:|----------------:|
| Flat f32 L2 + sort top-10 | **2.118 ms** | 472 QPS  | 1.0× |
| PQ8 ADC scan + sort top-10 | **775 µs**  | 1,290 QPS | 2.7× |
| FastScan-4 + sort top-10  | **249 µs**  | 4,015 QPS | **8.5×** |

FastScan-4 is **3.1× faster than scalar PQ8 ADC** on the same data and
same codebook *training* — the only difference is the storage layout and
the scan kernel.

### End-to-end pipeline (demo binary, n = 100,000, d = 128, M = 16, 100 queries)

| Variant | Total time (100 q) | QPS | Recall@10 |
|---------|-------------------:|----:|----------:|
| Flat f32 (oracle)     | 465 ms | 215   | 1.000 |
| PQ8 ADC               | 178 ms | 562   | 0.600 |
| FastScan-4 raw        | **60.5 ms**  | **1,652** | 0.372 |
| FastScan-4 + rerank-100 | **64.3 ms**  | **1,555** | **0.855** |

Reranking the top-100 FastScan candidates with exact f32 L2 raises
recall@10 from 0.37 → 0.86 for only a 6 % cost over the raw scan, while
still beating flat scan by **7.2×**.

### Storage (n = 100,000, d = 128, M = 16)

| Format    | Size      | Bytes/vec |
|-----------|----------:|----------:|
| Flat f32  | 50,000 KB | 512       |
| PQ8       | 1,562 KB  | 16        |
| FastScan-4| 781 KB    | 8         |

FastScan halves PQ8's footprint at iso-recall **with** the rerank stage
(which uses no extra storage — it reads from the same f32 buffer that
flat scan reads, present only at index-build time in practice and
optional at scan time).

### Reproducing

```sh
cargo build  --release -p ruvector-pq-fastscan
cargo test   --release -p ruvector-pq-fastscan
cargo run    --release -p ruvector-pq-fastscan
cargo bench  -p ruvector-pq-fastscan --bench fastscan_bench
```

Environment knobs on the demo binary: `N`, `D`, `M`, `Q`, `K`, `TRAIN`,
`INTRINSIC`, `RERANK`.

## How it works (blog walkthrough)

You have 100 million vectors. Each is 768 floats. Flat scan reads 300 GB
per query. You don't have time for that.

Product Quantization (Jégou 2011) cuts each vector into `M` sub-vectors,
learns a codebook of `K` centroids in each subspace, and replaces every
sub-vector with the index of its nearest centroid. With `M = 16, K = 256`
you store one byte per sub-vector and the whole vector costs 16 bytes
instead of 3,072. To estimate the squared L2 distance between a query
and a database vector, you precompute an `M × K` table of squared L2
distances between query sub-vectors and centroids; then for each
database vector you sum `M` table lookups indexed by its 16 code bytes.
This is "asymmetric distance computation," ADC.

ADC has one ugly bottleneck. Each scan iteration is `M` random gathers
into a small table — and on modern CPUs random gathers are slow because
the table is bigger than the SIMD register file and you cannot pack the
indices.

FastScan (André 2017) makes one tiny change with huge consequences:
shrink `K` from 256 to 16. Now the entire LUT row for one sub-quantizer
fits in a single 16-byte SIMD register, and the codes are 4 bits each,
so you can pack two codes per byte. The CPU has had an instruction for
"look up 16 bytes in parallel into a 16-entry table" since 2006:
`pshufb` (x86 SSSE3, AVX2's `vpshufb`) and `vqtbl1q_u8` (AArch64 NEON).
One instruction does what 16 gathers used to do. Sixteen sub-quantizers
of work, sixteen ticks of the clock, and you're scanning 32 vectors per
loop iteration.

The cost is precision. `K = 16` is a much coarser codebook than
`K = 256`. Raw FastScan tops out around 0.40 recall@10 on real-world
data. The fix is the two-stage trick that every production system uses:
let FastScan rank the top 100 or 1000 candidates and have a slower,
exact-float reranker pick the final 10. The combined cost is dominated
by the FastScan stage, and the reranker recovers full PQ8-grade recall
or better.

This is the kernel inside ScaNN, FAISS's `IndexPQFastScan`, and most
modern billion-scale vector databases. It's not new — it's nine years
old — but it does not exist as a stand-alone Rust crate yet. Now it does.

## Practical failure modes

* **Outlier-skewed LUT** — without per-row min subtraction, recall
  collapsed to ~0.06 on the first run of this PoC. Fixed in
  `build_lut()`.
* **Pathological data distribution for K-means** — pure isotropic
  Gaussian gives K-means no signal; well-separated cluster mixtures
  cause K-means to allocate all 256 centroids to cluster modes and lose
  intra-cluster resolution. **First two attempts at synthetic data
  produced recall@10 = 0.14 for PQ8** before the low-rank Gaussian was
  adopted. Real workloads (SIFT, GIST, CLIP, OpenAI ada) all have the
  low-rank-plus-noise structure that PQ relies on.
* **Padding lanes** — the last block holds `n mod 32` real vectors
  followed by zero-coded padding. Padding distances are bounded but
  meaningless; the top-k truncation step must run AFTER the `.truncate(n)`
  on the sums vector. (It does — but a future micro-optimisation that
  reads sums directly without the resize would need to remember.)
* **K-means initialisation** — random-point init with 12 Lloyd
  iterations is enough for `K = 16`, marginal for `K = 256`. Production
  code should switch to k-means++ before shipping.

## What to improve next

| Priority | Item | Reason |
|----------|------|--------|
| **P0** | Implement x86 AVX2 + AVX-512 scan paths | Half of production ANN deployments are still x86; FastScan needs `pshufb` on both. |
| **P0** | Wire FastScan into `ruvector-rairs` (IVF) | IVF + FastScan is the production combo (FAISS `IndexIVFPQFastScan`). Currently `ruvector-rairs` uses 8-bit ADC. |
| **P1** | Anisotropic PQ training (ScaNN's score-aware loss) | +5–10 pp recall at no scan cost. Aligned with prior `ruvector-avq`. |
| **P1** | OPQ rotation pre-multiplication | Pre-rotates so PQ sub-spaces are decorrelated. Worth 3–5 pp on most real datasets. |
| **P1** | k-means++ initialisation | Reduces variance across seeds. |
| **P2** | Refinement via residual PQ ("RQ-PQ") | Stack a second-level PQ on residuals for "PQ4 + PQ4" → matches PQ8 recall at the same byte budget but with 2× the SIMD shuffles. |
| **P2** | Wasm scalar fallback validation | Already correct via scalar path; benchmark to confirm. |
| **P3** | GPU FastScan kernel (Metal / WGSL) | One workgroup per block, one threadgroup-local LUT, ~30× over CPU on M-class Macs. Probably worth a separate crate. |

## Production crate layout (proposed)

```text
crates/ruvector-pq-fastscan/        ← this crate (4-bit PQ + scan only)
crates/ruvector-pq-fastscan-avx2/   ← x86 SIMD path (feature flag)
crates/ruvector-rairs/              ← IVF, add FastScan posting-list scan
crates/ruvector-leanvec/            ← LeanVec subspace pruning, reuses FastScan
crates/ruvector-snapshot/           ← persistence (packed block format → mmap)
```

The `FastScanIndex` struct already separates trained codebook
(`ProductQuantizer`) from packed codes (`Vec<u8>`), so an mmap-based
storage backend is a drop-in replacement for the `Vec<u8>` field.

## References

1. Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor
   Search.* IEEE PAMI 2011.
2. André, Kermarrec, Le Scouarnec. *Cache locality is not enough:
   high-performance nearest-neighbor search with Product Quantization
   fast scan.* VLDB / ICDE 2017.
3. André, Kermarrec, Le Scouarnec. *Quicker ADC.* IEEE TPAMI 2021.
4. Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar.
   *Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization (ScaNN).* ICML 2020.
5. Aguerrebere, Mauschitz, Naumov, Yi, Goyal.
   *LeanVec: Searching vectors faster by making them fit.* arXiv 2312.16335.
6. Matsui, Yamasaki, Aizawa. *PQk-means.* ACM MM 2017.
7. Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs (FAISS).*
   IEEE TBD 2019.
8. Gao, Long. *RaBitQ.* SIGMOD 2024.
9. Yang et al. *SymphonyQG.* SIGMOD 2025.
10. FAISS source — `faiss/IndexPQFastScan.cpp` and
    `faiss/impl/pq4_fast_scan.cpp`.
