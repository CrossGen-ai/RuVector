# ruvector 2026: Tribase Triangle-Inequality Pivot Pruning — Exact, High-Performance Rust Vector Search

> **Summary:** ruvector now ships exact triangle-inequality pivot pruning for IVF — 0.78% memory overhead, up to 2.44× QPS speedup, recall 1.000 preserved. Pure Rust, CPU-only, no quantization required.

## Introduction

[ruvector](https://github.com/ruvnet/RuVector) is a high-performance Rust vector database that already ships RaBitQ quantization, ACORN filtered search, multi-vector MUVERA, hyperbolic HNSW, and more. The 2026-06-15 nightly research branch adds **Tribase**: a triangle-inequality pivot pruning layer for IVF that is **provably exact** — pruning skips only vectors that *cannot* enter top-k. No recall tuning knob, no statistical fudge, just geometry.

The headline number: on a single Apple Silicon thread, Tribase delivers **up to 2.44× QPS** over standard IVF at **recall 1.000**, using just **4 bytes of overhead per vector** (one f32 residual).

## Features

- **Exact pruning.** Recall = 1.000, bit-for-bit identical to exhaustive IVF on the same probed lists.
- **0.78% memory overhead.** One f32 per vector (the distance from each vector to its IVF centroid).
- **CPU-only, pure Rust.** Two dependencies: `rand`, `rand_chacha`. No `unsafe`. No GPU. No FFI.
- **Bisect-and-sweep search.** Lists sorted by residual; bisect at `d(q,c)`; sweep outward with short-circuit on both halves.
- **Drop-in benchmark harness.** Reproducible `cargo run --release` numbers for Flat / IVF / Tribase on identical data.
- **Composable.** Geometry-only — combines with RaBitQ (probabilistic LB) and ACORN (predicate filters) without changes.

## Benefits

- **Recall is not a tuning parameter.** Many ANN systems trade recall for speed; Tribase is loss-free by construction.
- **Memory budget barely moves.** 4 bytes per vector vs. tens of bytes for quantized lower bounds.
- **No retraining.** Build-time cost is one extra distance per vector + one sort per cluster — no learned codebook, no PCA, no rotation.
- **Production-honest results.** Bench reports include a configuration where Tribase regresses (0.93×) — published rather than hidden, with a documented failure mode.

## Comparisons

| System | Memory overhead | Recall guarantee | Per-vector LB type | Build cost |
|---|---|---|---|---|
| **ruvector Tribase** | **~4 B/vec** | **exact = 1.000** | **geometric (triangle)** | k-means + sort |
| FAISS IVF (baseline) | 0 | exact = 1.000 | none | k-means |
| Milvus IVF_FLAT | 0 | exact = 1.000 | none | k-means |
| Qdrant HNSW (default) | graph links | tunable | none | graph build |
| Weaviate HNSW | graph links | tunable | none | graph build |
| Pinecone (default) | proprietary | tunable | none | proprietary |
| FAISS IVF_PQ | ~D/4 B/vec | probabilistic | quantization | k-means + PQ codebook |
| RaBitQ | ~D/8 B/vec | probabilistic | 1-bit signature | k-means + bit-pack |

The relevant comparison axis is **memory per vector for the pruning structure** and **whether recall is exact**. Tribase wins both: lowest memory among methods that prune, and recall is exact.

## Benchmarks

Hardware: Apple Silicon, Darwin 24.6.0 (arm64), single-threaded. Synthetic data: 50 Gaussian clusters, σ = 0.4 — mimics CLIP / SBERT / ada-002 cluster geometry. k = 10. Ground truth from exhaustive flat search.

| n      | dim | n_lists | nprobe | method      | QPS         | avg dist ops | recall    | p99 (µs) |
|--------|-----|---------|--------|-------------|-------------|--------------|-----------|----------|
| 10 000 | 64  | 64      | 8      | flat        |     4 964   |   10 000     |   1.000   |   256.5  |
| 10 000 | 64  | 64      | 8      | ivf         |    27 515   |    1 302     |   1.000   |   120.8  |
| 10 000 | 64  | 64      | 8      | **tribase** | **54 708**  | **638**      | **1.000** |  **60.8**|
| 20 000 | 128 | 128     | 16     | ivf         |     5 449   |    2 561     |   1.000   |   444.1  |
| 20 000 | 128 | 128     | 16     | **tribase** |  **7 629**  | **2 021**    | **1.000** | **348.1**|
| 50 000 | 128 | 256     | 24     | ivf         |     2 207   |    4 767     |   1.000   |  1 078.4 |
| 50 000 | 128 | 256     | 24     | tribase     |     2 062   |    3 956     |   1.000   |  1 121.0 |
| 20 000 | 128 | 128     | 32     | ivf         |     2 414   |    4 862     |   1.000   |  1 467.4 |
| 20 000 | 128 | 128     | 32     | **tribase** |  **5 897**  | **2 561**    | **1.000** | **941.8**|

**Takeaways**

- **17 % – 51 % fewer distance computations** vs. standard IVF across all configurations.
- **Up to 2.44× QPS speedup** at the same recall.
- **0.78 % memory overhead** (one f32 per vector — 78 KiB on 20 K vectors of dim 128).
- **One regression** on `n = 50 000, nprobe = 24`: 17 % distance saving did not translate to wall-time because bisect/sqrt overhead dominated. Honest reporting matters.

## Optimizations

Already in the prototype:

- Squared L2 internally; `sqrt` is only called on `tau` and on `d(q, centroid)`, **never per vector**.
- 4-way unrolled scalar distance kernel.
- Hand-rolled max-heap over `(f32, u32)` — avoids the `BinaryHeap<OrderedFloat>` wrapper cost.
- Parallel `Vec<u32>` / `Vec<f32>` layout for ids and residuals — cache-friendly sweep.
- Deterministic via seeded `ChaCha8Rng` — reproducible benchmarks.

Next, planned in the research doc:

- **SIMD distance kernels** (AVX2 / NEON) — ~4× the inner loop.
- **Multi-pivot Tribase** — multiple anchors per cell, take the tightest LB. Liu et al. 2024 report another ~1.5–2× distance-op reduction.
- **Composition with RaBitQ** — `max(triangle LB, quantized LB)` prunes harder than either alone.
- **`rayon::par_iter` over probed lists** — embarrassingly parallel.
- **Cosine variant** for inner-product workloads (`r(x) = 1 - <x, c>` for unit vectors).

## Get Started

The implementation lives on the **CrossGen-ai fork** of ruvector, on the nightly research branch:

```
https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-06-15-tribase-pivot-pruning
```

Upstream project: [github.com/ruvnet/RuVector](https://github.com/ruvnet/RuVector). Nightly research branches like this one live on the CrossGen-ai fork only — they are not opened as upstream PRs (they are pure exploration; promotions to upstream happen through a separate ADR-graduation workflow).

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-06-15-tribase-pivot-pruning
cargo test --release -p ruvector-tribase
cargo run --release -p ruvector-tribase --example bench_pruning
```

Read the full design + SOTA survey in `docs/research/nightly/2026-06-15-tribase-pivot-pruning/README.md` and the architecture decision in `docs/adr/ADR-253-tribase-pivot-pruning.md`.

---

**Keywords:** ruvector, vector search, ANN, approximate nearest neighbor, IVF, triangle inequality, Tribase, SIGMOD 2024, exact pruning, Rust vector database, high-performance ANN, RaBitQ, FAISS alternative, Milvus alternative, Qdrant alternative, Weaviate alternative, Pinecone alternative, CPU-only ANN, k-means pivot pruning, recall 1.0 ANN, low-memory vector search, embedding search, semantic search Rust.
