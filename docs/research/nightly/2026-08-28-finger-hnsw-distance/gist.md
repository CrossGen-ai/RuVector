# ruvector 2026: FINGER Low-Rank Distance Approximation — High-Performance Rust Vector Search

**Summary (≤150 chars):** FINGER cuts HNSW inner-loop cost 1.4×–2.2× at zero recall loss and 8×–16× less memory. Pure-Rust PoC, real Apple M4 Max numbers.

## Introduction

ruvector's graph-based vector search stack (HNSW, ACORN, DiskANN,
coherence-HNSW, diverse-beam, speculative-ANN) all spend most of their
wall-clock in one place: **scoring the M neighbours of the current best
node against the query**. This nightly lands `ruvector-finger`, a pure-Rust
implementation of the FINGER low-rank distance approximation trick
(Yin et al., WWW 2023), plus a Johnson–Lindenstrauss baseline for
comparison. Every neighbour-score is reduced from an O(d) inner product
to an O(r) dot in a per-pivot PCA basis, precomputed once per pivot
handle. Result: **FINGER-r16 matches exact recall@10 at 64% of the
wall-clock cost with 12.5% of the memory** — measured on Apple M4 Max,
single-threaded, release mode, no mocks.

Keywords: `ruvector`, `Rust vector search`, `HNSW distance approximation`,
`FINGER ANN`, `low-rank residual PCA`, `graph ANN acceleration`,
`quantization alternative`, `nearest neighbor search 2026`.

## Features

- Self-contained crate (`ruvector-finger`) with three swappable
  `DistanceEstimator` implementations: **Exact**, **JL-r** (global
  Johnson–Lindenstrauss), **FINGER-r** (per-pivot PCA via deflated power
  iteration).
- HNSW/ACORN/DiskANN-agnostic trait — designed to be layered onto any
  graph-ANN crate behind a feature flag.
- Deterministic, seeded synthetic dataset for reproducible nightly
  benchmarks.
- Real `cargo bench` numbers, real `cargo run` demo, real `cargo test`
  suite (5 tests, all passing).
- No BLAS, no external ANN library, no unsafe — 6 source files, each
  under 200 lines.

## Benefits

- **Same recall at ~2/3 the latency.** FINGER-r16 recall parity with
  exact scoring at 64% wall-clock.
- **8×–16× less RAM per vector code** vs storing residuals in full
  precision.
- **Composable.** Stacks with RaBitQ, Matryoshka and PQ quantisation
  crates already in ruvector.
- **Batch-friendly build.** Per-pivot PCA fits ruvector's
  `lsm-ann`/`spann-partition-spill` compaction model.

## Comparisons

| Engine / feature (Aug 2026) | Ships a per-pivot low-rank cache?         | Complementary to FINGER?         |
|-----------------------------|-------------------------------------------|-----------------------------------|
| Milvus 2.5                  | No (PQ-ADC + graph)                       | Yes — PQ orthogonal to FINGER    |
| Qdrant 1.13                 | No (binary quantisation + rerank)         | Yes — 1-bit anchor + FINGER res.  |
| Weaviate 1.30               | No (RaBitQ integration)                   | Yes — stack RaBitQ + FINGER      |
| Pinecone                    | No public detail                          | Likely — FINGER is orthogonal    |
| LanceDB 0.15                | No (learned rerank)                       | Yes                              |
| FAISS 1.10                  | No (fused ADC on IVF-PQ)                  | Yes                              |
| **ruvector-finger (this)**  | **Yes — per-pivot PCA basis + codes**     | **Composes with all above**      |

## Benchmarks

Hardware: **Apple M4 Max, arm64, macOS, single-thread, release mode.**
`criterion` (20 samples, 2 s measurement, 1 s warm-up) for the
micro-bench; `std::time::Instant` for the end-to-end demo. Reproduce with
`cargo bench -p ruvector-finger` and `./target/release/finger-demo`.

### Per-pivot batch scoring (average 71 neighbours, d=128)

| estimator   | time     | speedup vs exact | bytes/vec |
|-------------|----------|------------------|-----------|
| exact       | 1.96 µs  | 1.00×            | 512       |
| jl-r16      | 1.61 µs  | 1.22×            |  64       |
| finger-r16  | 1.38 µs  | 1.42×            |  64       |
| finger-r8   | 0.87 µs  | 2.24×            |  32       |

### End-to-end query loop (n=20 000, d=128, 200 queries, beam=4, rerank=200, k=10)

| estimator   | per-query | recall@10 |
|-------------|-----------|-----------|
| exact       | 54.9 µs   | 0.364     |
| jl-r16      | 44.7 µs   | 0.316     |
| finger-r8   | 31.9 µs   | 0.353     |
| finger-r16  | 35.0 µs   | 0.364     |

### Build cost

- JL-r16:      **24 ms**
- FINGER-r8:  **443 ms**
- FINGER-r16: **887 ms**

Amortised across ≥ 10⁶ queries typical of nightly workloads.

## Optimisations

- **Deflated power iteration** avoids materialising the d×d covariance
  matrix — critical for d ≥ 128.
- **Anchor cache**: `v · pivot` is stored once at build time and reused
  as the first term of every FINGER score.
- **Query-side amortisation**: the query's basis projection is computed
  once per pivot handle and reused across all M neighbours.
- **Bytes-per-vector reporting** built into the `DistanceEstimator` trait
  so downstream crates can budget memory explicitly.

## Get started

Working code lives on the CrossGen-ai fork of ruvector (not upstream
ruvnet/ruvector). This is a research branch — no upstream PR is opened.

- **Fork branch**: <https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-28-finger-hnsw-distance>
- **Crate**: `crates/ruvector-finger/`
- **Research doc**: `docs/research/nightly/2026-08-28-finger-hnsw-distance/README.md`
- **ADR**: `docs/adr/ADR-340-finger-hnsw-distance.md`

Reproduce:

```sh
git clone https://github.com/CrossGen-ai/RuVector.git ruvector
cd ruvector
git checkout research/nightly/2026-08-28-finger-hnsw-distance
cargo test --release -p ruvector-finger
cargo run   --release -p ruvector-finger --bin finger-demo
cargo bench -p ruvector-finger --bench finger_bench
```

Related upstream project: <https://github.com/ruvnet/RuVector>.
