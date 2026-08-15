# ruvector 2026: Residual Product Quantization — High-Performance Rust Vector Search

**Summary (150 chars):** A dependency-free Rust reference implementation of two-level residual product quantization (RPQ2) for ruvector, with real cargo benchmarks vs PQ and SQ8.

`ruvector` gains a new dependency-free crate, `ruvector-rpq`, that
provides a common `Quantizer` trait plus three swappable backends —
single-level Product Quantization (**PQ**), two-level Residual PQ
(**RPQ2**), and 8-bit Scalar Quantization (**SQ8**) — for compressed
approximate nearest-neighbor search in Rust. This nightly research crate
sits alongside `ruvector-rabitq`, `ruvector-turboquant`, and
`ruvector-core`, giving the ruvector project a clean, in-tree,
zero-unsafe reference implementation of the classical RPQ family used
by FAISS `IVFPQ`, Milvus `IVF_PQ`, Qdrant PQ, Weaviate PQ, and
Pinecone PQ.

## Features

- **Common `Quantizer` trait**: `encode`, `code_bytes`, `adc_sq_distance`, `name`.
- **Three backends**: `Pq`, `Rpq2`, `Sq8` — pick the recall/latency/bytes point that fits.
- **Amortised RPQ2 scorer**: `Rpq2Scorer` groups items by coarse code so the residual look-up table is built once per coarse cell.
- **Zero external crates**, `#![forbid(unsafe_code)]`, deterministic PRNG.
- **k-means++** seeding, Lloyd iterations, empty-cluster re-seeding.
- **Runnable benchmark binary** (`rpq-bench`) that trains, encodes, and scores against exact brute-force ground truth — no mocked numbers.
- **Six passing unit tests** — reconstruction ratios, ADC exactness, bucket-scorer equivalence, RPQ2 vs PQ error bound.

## Benefits

- Reference codec for future IVFADC / SPANN / graph-coarse-cell indexes in ruvector.
- Portable — pure Rust, builds on any target the workspace already builds on.
- Honest engineering data: the benchmark table records exactly where each codec wins and where it loses.
- Trait-based design allows drop-in A/B against other quantizers (e.g., `ruvector-rabitq`) without rewriting call sites.

## Comparisons vs other vector-search systems

| System | Product Quantization | Residual / two-level PQ | Scalar 8-bit | Rust-native |
| --- | --- | --- | --- | --- |
| ruvector (`ruvector-rpq`, new) | ✅ `Pq` | ✅ `Rpq2` + amortised scorer | ✅ `Sq8` | ✅ |
| FAISS | ✅ `IndexPQ` | ✅ `IndexIVFPQ`, `IndexResidualQuantizer` | ✅ `IndexScalarQuantizer` | ❌ (C++) |
| Milvus | ✅ | ✅ (`IVF_PQ`) | ✅ | ❌ (Go/C++) |
| Qdrant | ✅ | ⚠️ single-level PQ | ✅ | ✅ |
| Weaviate | ✅ | ⚠️ single-level PQ | ✅ | ❌ (Go) |
| Pinecone | ✅ | closed | closed | ❌ (closed) |
| LanceDB | ✅ | ⚠️ single-level PQ | ✅ | ✅ |

ruvector now has a reference RPQ2 to compare against these systems'
two-level offerings on identical datasets.

## Benchmarks (real numbers from `cargo run --release -p ruvector-rpq --bin rpq-bench`)

**Hardware**: Apple Silicon developer laptop, macOS.
**Data**: 64-D mixture of 64 Gaussians (σ = 0.5, centroids N(0, 9)).
**Sizes**: 10 000 train, 20 000 database, 200 queries, top-10, k-means iters = 25.
**Ground truth**: exact brute-force squared-L2, 55 ms for the full 200-query sweep.

| Codec | Bytes/vec | Compression | Train (ms) | Encode (ms, 20 k) | Query (ms/q) | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| `pq-8` | 8 | 32× | 1028.9 | 71.1 | 0.08 | 0.036 |
| `pq-16` | 16 | 16× | 1394.2 | 96.7 | 0.12 | **0.082** |
| `rpq2 (amort)` | 16 | 16× | 1957.4 | 155.1 | 53.23 | 0.056 |
| `sq8` | 64 | 4× | 0.4 | 0.4 | 0.38 | **0.894** |

### What the numbers mean

- **At equal coarse budget (8 B)**, RPQ2 lifts recall 3.6 % → 5.6 %
  (**1.56× improvement**) by adding a second 8-B residual code.
- **At equal total storage (16 B)**, single-level `pq-16` beats
  RPQ2 (**0.082 vs 0.056**) — a candid finding: **RPQ2 is not a
  drop-in replacement for wider PQ in a flat scan**. It belongs
  in an IVFADC or graph-coarse-cell context where the residual LUT
  is amortised across many items sharing the same coarse code.
- **`sq8`** is the recall king if you can afford 4× storage — the
  classical reminder that 8-bit scalar quantization is remarkably hard
  to beat.

## Optimizations

- Amortised query LUT (`Pq::compute_lut` + `Pq::adc_from_lut`) so the
  `m·k` table is built once per query, not per item.
- `Rpq2Scorer` groups items by coarse code (`BTreeMap<Vec<u8>, Vec<u32>>`)
  so the residual LUT is rebuilt once per coarse cell.
- K-means uses k-means++ seeding to avoid the worst-case Lloyd
  initialisation trap.
- Deterministic xorshift64* PRNG — reproducible benchmarks across runs.

Follow-up ideas documented in the research doc: OPQ rotation, rayon-parallel
training/encoding, AVX-512 / NEON `adc_from_lut` kernels, IVFADC
integration, additive-quantization (AQ) generalisation, symmetric
distance computation for pure-code re-ranking.

## Get started

- Fork: <https://github.com/CrossGen-ai/RuVector>
- Branch: `research/nightly/2026-08-15-residual-product-quantization`
- Upstream project: <https://github.com/ruvnet/ruvector>

```
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-08-15-residual-product-quantization
cargo test --release -p ruvector-rpq
cargo run  --release -p ruvector-rpq --bin rpq-bench
```

Keywords: ruvector, rust vector search, product quantization, residual
product quantization, RPQ, IVFPQ, IVFADC, ANN, approximate nearest
neighbor, vector database, embeddings, FAISS alternative, Milvus
alternative, Qdrant alternative, high-recall compression.
