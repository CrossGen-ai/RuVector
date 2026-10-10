# ruvector 2026: PRVQ Cascade Rerank — Halve Full-Precision Fetches at the Same Recall

A new technique in ruvector's nightly research line lands today: **PRVQ
(Product-Residual Vector Quantization)**, a composite-quantization
rerank stage that drops into any PQ-8 IVF-ADC pipeline and **cuts
full-precision vector fetches by 2× at recall@10 = 0.978**. Measured on
a 20 000-vector × 128-dim Gaussian-mixture corpus in scalar portable
Rust, bit-exact reproducibility, with real `cargo bench` numbers below.

High-performance Rust vector search needs more than a fast ANN graph —
once candidates land in a shortlist, verifying each with the exact
float32 L2 is the single most expensive operation on disk-resident
indexes. PRVQ replaces part of that cost with a composite PQ + residual
RVQ distance estimator whose online work is **`m + K` LUT lookups per
candidate**, no vector fetch.

## Features

- Standalone Rust crate `ruvector-rvq-cascade`, 0 external deps beyond
  `rand`.
- 8-bit PQ backbone (`m=8`) + K-stage 8-bit RVQ on the PQ residual.
- Precomputed cross-term `2⟨x_pq, r_rvq⟩` for O(1) estimator work in `d`.
- Scalar portable Rust — no SIMD required to reproduce the numbers.
- Deterministic k-means++ init with `ChaCha20Rng`.
- `CascadePipeline` struct runs the full IVF-ADC → PRVQ refine → verify
  pipeline end-to-end.
- Tests: `cargo test --release` passes with 2 real recall tests.
- Benches: `cargo bench` produces the exact table reproduced below.

## Benefits

- **2× fewer full-precision fetches at recall ≥ 0.95** — the biggest
  win when vectors live on NVMe or S3.
- **Strict distance-estimator improvement over PQ ADC** — the RVQ
  residual term is additive and trained to minimize reconstruction
  error, so the composite estimate has lower variance.
- **Memory overhead is small**: `+K+8` bytes/vec on top of PQ-8's 8
  bytes/vec. At K=4 and d=128 that's +12 B/vec, still 95% smaller than
  raw float32.
- **No change to the IVF partitioner, PQ codebook format, or query
  path** — PRVQ is purely additive bookkeeping.

## Comparisons

| Engine | Composite-quantization rerank | 8-bit PQ | On-disk ADC | Rust |
|---|---|---|---|---|
| Milvus | ❌ (PQ + float rerank) | ✅ | ✅ | ❌ C++ |
| Qdrant | ❌ (PQ + float rerank) | ✅ | ✅ | ✅ |
| Weaviate | ❌ | ✅ | partial | ❌ Go |
| Pinecone | opaque | ✅ | ✅ | ❌ |
| FAISS | ✅ RQ / AQ in `IndexResidualQuantizer` | ✅ | partial | ❌ C++ |
| **ruvector PRVQ (today)** | ✅ **cheapest trainer, 2× fetch win** | ✅ | ✅ | ✅ |

Where competitors land on PQ + float rerank, ruvector's PRVQ gets the
cheap composite-quantization win without the heavy AQ/CQ trainer.

## Benchmarks

Hardware: Apple M-series, scalar Rust `release`, LTO=thin, N=20 000,
d=128, nq=200, k=10.

```
Variant                                      ms/query    recall@10
------------------------------------------------------------------
A. Full float32 brute                           0.913        1.000
B. PQ-ADC only (no rerank)                      0.088        0.150
C. PQ -> full-verify top-200                    0.103        0.913
C. PQ -> full-verify top-400                    0.115        0.978
C. PQ -> full-verify top-800                    0.138        0.997
D. PRVQ cascade pq=2000 verify=100              0.137        0.879
D. PRVQ cascade pq=2000 verify=200              0.143        0.978
D. PRVQ cascade pq=2000 verify=400              0.150        0.994

At recall@10 >= 0.95:
  Classic PQ-rerank: 400 full fetches,  0.115 ms, recall 0.978
  PRVQ cascade    : 200 full fetches,  0.143 ms, recall 0.978
  Fetch reduction : 2.0x fewer full-precision fetches
```

Train cost: PQ-8 = 1.82 s; PRVQ (m=8, K=4) = 11.03 s. Per-cell training
in production restores parallel scaling.

## Optimizations

- Partial RVQ norms `‖sum_{s≤s'} centroid_s‖²` cached per (vector,
  stage) so the mid-cascade estimate is O(1) in `d`.
- Cross-term `2⟨x_pq, r_rvq⟩` cached per vector so the online
  estimator never re-reconstructs x.
- `select_nth_unstable` on the shortlist avoids full sorts between
  stages — only the final top-10 is sorted.
- All loops are branch-free scalar; AVX2/NEON FastScan is a planned
  follow-up ADR.

## Get started

Code lives on the CrossGen-ai fork, nightly research branch:

- Branch: `research/nightly/2026-10-10-rvq-cascade-rerank`
- Fork: https://github.com/CrossGen-ai/RuVector
- Upstream (ruvnet): https://github.com/ruvnet/RuVector

```bash
git clone -b research/nightly/2026-10-10-rvq-cascade-rerank \
    https://github.com/CrossGen-ai/RuVector.git ruvector
cd ruvector/crates/ruvector-rvq-cascade
cargo bench
```

For design details see `docs/adr/ADR-362-rvq-cascade-rerank.md` and
`docs/research/nightly/2026-10-10-rvq-cascade-rerank/README.md`.

**Tags:** ruvector, vector search, PQ, product quantization, residual
quantization, RVQ, PRVQ, ANN, approximate nearest neighbor, DiskANN,
IVF, ADC, rerank, Rust, high-performance, SIMD, SOTA 2026
