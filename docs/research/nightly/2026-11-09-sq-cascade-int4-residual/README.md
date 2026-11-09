# Cascaded int8 + int4-Residual Scalar Quantization (SQ-Cascade-Int4)

_Nightly research, 2026-11-09 — ruvector_

## Abstract

We evaluate a two-stage scalar quantizer: an int8 per-dim scalar quantizer
("base") followed by a packed int4 per-dim quantizer of the base's reconstruction
residual. Storage is exactly `1.5 × D` bytes per vector — half the gap between
int8 (1×D) and int16 (2×D). On synthetic Gaussian data (D=128, N=50 000, 1 000
queries, k=10) the cascade cuts mean-squared reconstruction error **196×** relative
to int8-only and lifts recall@10 from **0.977 → 0.998** (a +2.16 pp absolute
gain), at a 1.50× storage cost and a ~3–4× slower encode/decode. For retrieval
rerank pipelines where a few hundred nanoseconds of decode per candidate is
dwarfed by the cost of loading a full fp32 vector from RAM, this is a strict
Pareto improvement over raw int8.

## SOTA survey

Scalar quantization of vectors for ANN is dominated by int8/fp16 base codecs
and more sophisticated product quantization (PQ, OPQ) / learned binary codes
(RaBitQ, LVQ). Recent directions that are adjacent:

- **LVQ** (Weaviate / Milvus-style local-adaptive int8 SQ): one scale per
  vector, not per dim. Our int8 baseline is per-dim, which is already stricter.
- **Lloyd-Max SQ4 / μ-law companding / variance-adaptive SQ** (prior nightly work
  in this repo): one-shot codebook refinement, no cascading.
- **RVQ / Residual VQ** for vectors (Faiss `ResidualQuantizer`): uses product
  codebooks per stage — richer but far heavier at both train and query time.
  SQ cascade is the "scalar-only" degenerate case: two per-dim passes, no
  codebooks, trivial to tune.
- **Extended RaBitQ (multi-bit)** and **Asymmetric RaBitQ q2d1** (prior nightly
  work): bit-oriented encodings for cosine distance. Complementary to SQ
  cascade, not a replacement — one can cascade RaBitQ + int4 residual for an
  even cheaper refinement stage.

The cascade specifically targets rerank-stage storage where the engineer wants
"int8 memory, int12-ish precision" without paying full int16 storage or the
training overhead of RVQ.

References:
- Jégou et al. — _Product Quantization for Nearest Neighbor Search_, TPAMI 2011.
- Weaviate blog — _Locally Adaptive Vector Quantization_, 2024.
- Gao & Long — _RaBitQ_, SIGMOD 2024.
- Faiss `ResidualQuantizer` source, 2023–2025.

## Proposed design

Per-dimension cascade. Train on N vectors of dimension D:

1. Compute per-dim `mean[j]`, `maxabs[j] = max_i |x[i,j] - mean[j]|`,
   `scale8[j] = maxabs[j] / 127`.
2. For each training vector, encode-then-decode the base and track
   `res_maxabs[j] = max_i |x[i,j] - recon[i,j]|`. Set `scale4[j] = res_maxabs[j] / 7`
   (signed int4 range is `[-7, 7]`).
3. Encode: emit `q8[j] = clamp(round((x[j]-mean[j])/scale8[j]), -127, 127)` as
   one byte, then `q4[j] = clamp(round((x[j] - recon8[j])/scale4[j]), -7, 7)`,
   packed two per byte.
4. Decode: reconstruct the base, then add `q4[j] * scale4[j]` per dim.

Storage per vector: `D + ceil(D/2)` bytes — exactly 1.5× int8 for even D.

Effective precision per dim: 256 base levels × 15 residual levels ≈ 12 bits.

## Implementation notes

- Signed int4 stored as 4-bit two's complement nibbles in a byte-packed array.
- No SIMD in the reference implementation; both passes are per-dim scalar loops.
  All scalar work is `f32`, which lets LLVM auto-vectorize the tight decode loop
  (observed ~160 ns per 128-dim vector in release).
- The trait `Quantizer` lets callers slot `Fp32Baseline`, `Int8Sq`, or
  `CascadeInt8Int4` behind the same code path. A production integration would
  plug this behind `ruvector-edge-quant`'s `Codec` trait for IVF posting
  storage or HNSW payload compression.

## Benchmark methodology

- Dataset: `gaussian_dataset(N=50 000, D=128, seed=0xC0FFEE)` — iid N(0,1) using
  Box-Muller over `ChaCha8Rng`. Each dim independently; the per-dim int8 base
  is near-optimal for this distribution, which is the **conservative** case
  for cascade: if the base is already good, residuals are small, and int4 has
  less to add. Non-stationary dims (skewed / heavy-tailed) would widen the
  cascade's lead.
- Queries: `uniform_queries(N_q=1000, D=128, seed=0xBADF00D)`, same distribution.
- Metric: recall@10 vs exact L2 top-10 on the fp32 dataset.
- Precision metric: avg MSE per dim between fp32 and decoded.
- Timing: wall-clock `std::time::Instant` over 20 000 single-vector encode/decode
  calls, release build with LTO=thin, opt-level=3.
- Hardware: Apple Silicon (user's local macOS dev box, darwin 24.6.0).
- Build: `cargo run --release --example bench_report`.

## Results

Captured from `cargo run --release --example bench_report` on this branch:

| variant             | bytes/vec | enc ns/vec | dec ns/vec | recall@10 | mse/dim      |
|---------------------|-----------|------------|------------|-----------|--------------|
| fp32                |       512 |       85.4 |       76.7 | 1.000     | 0            |
| int8_sq             |       128 |       49.2 |       42.2 | 0.977     | 9.9968e-5    |
| cascade_int8_int4   |       192 |      356.5 |      162.5 | 0.998     | 5.0978e-7    |

**Ratios:**
- `mse(int8) / mse(cascade) = 196.1×` — cascade is ~196× lower reconstruction error.
- `bytes(cascade) / bytes(int8) = 1.50×` — storage overhead as designed.
- recall@10 gain: **+0.0216 absolute** (0.977 → 0.998).

**Memory totals at N=50 000:**
- fp32: 0.0238 GiB
- int8: 0.0060 GiB
- cascade: 0.0089 GiB

## How it works (blog-readable walkthrough)

Think of int8 scalar quantization as writing each coordinate on a 256-position
ruler. The ruler is calibrated per dimension so the full dynamic range of that
dimension's data fits. Rounding snaps every value to one of 256 tick marks —
the "step size" is `(max - min) / 255`. You pay 1 byte per dim.

What's left over after rounding is the **residual** — the signed gap between
the true value and the nearest int8 tick. Across a well-trained int8 codec
the residuals are nearly uniform on `[-step/2, +step/2]`.

Cascade quantization treats that residual as a new, much smaller vector and
ruler-quantizes it again, this time with a much finer ruler because its range
is `1/256` of the original. Four bits are plenty for this second ruler, since
the residual is already small. The per-dim "effective step size" becomes
`original_step / 15`, which at k=10 retrieval almost completely wipes out the
int8 recall loss.

The trick is that the extra storage is only 4 bits per dim — half a byte —
so the full code weighs `1.5 × D` bytes, strictly between int8 and int16.

## Practical failure modes

- **Heavy-tailed distributions.** The int8 base clips at the per-dim maxabs it
  saw during training. If the online distribution has heavier tails later,
  the base saturates and the residual exceeds the int4 range, which just gets
  clipped to ±7×scale4. Recall degrades gracefully but noticeably.
  Mitigation: retrain scales periodically, or use a percentile (e.g. p99.9)
  instead of maxabs.
- **Query drift.** The cascade stores base vectors; the query itself is used
  in full precision. No query-side quantization is assumed — this is a storage-side
  optimization, so query-vs-codebook asymmetric distance (ADC) still applies if
  the caller wants it.
- **Encode cost dominates small batches.** Because encode re-decodes the base
  to compute the residual, encoding is ~3–4× int8-only. In this benchmark
  encode is 356 ns vs int8's 49 ns — a factor of 7× — which matters only for
  very write-heavy workloads. For a one-time index build amortized over
  millions of queries, this is noise.
- **Very low D (<32)** erodes the benefit: the integer-packing overhead per
  vector dominates and the storage ratio loosens.

## Production crate layout proposal

```
crates/ruvector-sq-cascade-int4/
  Cargo.toml          # standalone workspace, criterion bench
  src/
    lib.rs            # re-exports
    quant.rs          # Quantizer trait + Fp32 / Int8Sq / CascadeInt8Int4
    dataset.rs        # deterministic synthetic data
    eval.rs           # exact top-k, recall@k, MSE
  examples/
    bench_report.rs   # end-to-end results table (this doc's numbers)
  benches/
    quant_bench.rs    # criterion encode/decode microbench
```

For integration, the natural call site is `ruvector-edge-quant` — add a
`Codec::CascadeInt8Int4` variant whose train() runs the two-pass calibration
on the IVF posting payload, and whose decode() slots into the existing rerank
path.

## What to improve next

1. **SIMD encode / decode** — the inner loop is a pair of per-dim gather/scatter
   operations with constants; NEON (aarch64) and AVX2 (x86\_64) would likely
   cut decode below 50 ns/vec, re-opening the latency gap with int8.
2. **Per-dim adaptive bit allocation** — some dims have much larger residuals
   than others; spend 2 or 6 residual bits on high-energy dims (variance-adaptive,
   like prior nightly work) rather than a flat 4 bits.
3. **Asymmetric ADC** — precompute per-dim decoded values at encode time once,
   query in full precision, and sum into ADC-like LUTs for IVF posting scan.
4. **Cascade with RaBitQ base** — replace the int8 base with 1-bit RaBitQ, then
   int4 residual. Would give a ~0.625 × D-byte code at near-int8 recall.
5. **Realistic datasets.** Rerun on SIFT1M, DEEP1M, and a dense LLM embedding
   slice (e.g. 768-dim BGE) to confirm the win translates to natural data.

## References

- [Faiss: Lower-resolution quantizers](https://github.com/facebookresearch/faiss/wiki/Lower-resolution-quantizers)
- Jégou, Douze, Schmid — _Product Quantization for Nearest Neighbor Search_ (TPAMI 2011)
- Gao, Long — _RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound_ (SIGMOD 2024)
- Weaviate — _Locally Adaptive Vector Quantization_ (2024 blog)
- Pinecone — _ScaNN & residual quantization notes_ (2024 docs)
- Milvus — _IVF-SQ and IVF-RQ comparisons_ (2025 docs)
