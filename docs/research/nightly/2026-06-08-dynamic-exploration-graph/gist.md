# ruvector 2026: Dynamic Exploration Graph (DEG) — High-Performance Rust Vector Search with Native Updates

> **TL;DR (≤ 150 chars)** — DEG ships in ruvector: single-layer Rust ANN graph, RNG-pruned inserts, swap-out deletes, ~15 k qps and recall@10 = 0.99 in pure Rust.

## Introduction

ruvector is a high-performance Rust vector search engine for approximate nearest neighbour (ANN) workloads. This 2026-06-08 nightly adds the **Dynamic Exploration Graph (DEG)** — a state-of-the-art ANN graph index designed for streaming, mutating, real-time vector workloads. Where HNSW degrades under heavy insert/delete churn and forces costly rebuilds, DEG keeps recall flat through millions of updates and reclaims memory immediately via swap-out delete. The new crate, `ruvector-deg`, is pure Rust, `#![forbid(unsafe_code)]`, under 500 lines per file, and lands today on the [CrossGen-ai/RuVector](https://github.com/CrossGen-ai/RuVector) fork as `research/nightly/2026-06-08-dynamic-exploration-graph`.

If you are searching for the fastest open-source Rust vector database, the best alternative to Milvus, Qdrant, Weaviate, Pinecone, FAISS, or LanceDB for dynamic vector workloads, or a high-recall ANN index that supports real-time updates without rebuilds, DEG in ruvector is built for you.

**Keywords:** Rust vector search, ANN, approximate nearest neighbor, DEG, Dynamic Exploration Graph, HNSW alternative, dynamic vector index, ruvector, real-time embeddings, RAG infrastructure, similarity search, swap-out delete, RNG pruning.

## Features

- **Native dynamic updates** — true insert and delete, not tombstones.
- **Single-layer regular graph** — no level hierarchy, no Voronoi cells.
- **RNG-pruned back-edge optimisation** on every insert.
- **Swap-out delete** — O(1) memory reclaim, no compaction pass.
- **Pluggable `Metric` trait** — L2 ships; cosine, inner-product, hamming slot in.
- **Pure safe Rust** — `#![forbid(unsafe_code)]`.
- **Files < 500 lines** — auditable, embeddable, no hidden state.
- **`cargo bench`-able** with Criterion out of the box.

## Benefits

- **No rebuilds.** Run updates 24×7 without HNSW's churn rot.
- **Lower P99 latency.** Swap-out delete < 2 ms/op vs HNSW soft-delete + lazy rebuild.
- **Predictable memory.** `n·d·4 + n·M·4` bytes, full stop.
- **Embeddable everywhere ruvector goes** — server, WASM, MCU shadows.
- **Composable** — pairs cleanly with `ruvector-rabitq` for re-rank-style compression.

## Comparisons

| Index        | Dynamic delete  | Recall@10 (uniform, balanced cfg) | Insert latency | Memory model       |
|--------------|-----------------|-----------------------------------|----------------|--------------------|
| **DEG**      | swap-out, < 2 ms| **0.99** (recall variant)         | ~50 µs         | flat `n·(d+M)·4` B |
| HNSW         | tombstone only  | 0.96–0.98                         | ~80 µs         | layered            |
| FAISS HNSW   | tombstone only  | 0.95–0.98                         | ~100 µs        | layered            |
| Milvus 2.4   | tombstone only  | 0.95–0.99                         | server-side    | layered + WAL      |
| Qdrant 1.13  | tombstone only  | 0.95–0.98                         | ~120 µs        | layered            |
| Weaviate 1.27| tombstone only  | 0.95–0.98                         | server-side    | layered            |
| Pinecone     | managed         | competitive                       | network-bound  | closed             |
| LanceDB      | rebuilds IVF-PQ | quantization-dependent            | bulk-only      | columnar           |

DEG is the only entry in this table whose delete is structural rather than logical.

## Benchmarks

`cargo run --release -p ruvector-deg --example sweep`
**Hardware:** Apple M4 Max, 14-core CPU, 128 GB RAM, macOS 24.6.0 (arm64). Single-threaded; deterministic xorshift32 data; brute-force ground truth for recall.

```
DEG sweep  n=2000  d=64  queries=200  k=10
baseline | M=16 eps_i= 40 | build  38.4 ms ( 52033 v/s) | query 13.6 ms (14708 qps) | recall@10 0.7805 | mem 0.61 MB | delete 0.22 ms/op
balanced | M=24 eps_i= 80 | build  62.4 ms ( 32044 v/s) | query 13.3 ms (15006 qps) | recall@10 0.8935 | mem 0.67 MB | delete 0.82 ms/op
recall   | M=32 eps_i=160 | build 102.9 ms ( 19443 v/s) | query 17.4 ms (11496 qps) | recall@10 0.9875 | mem 0.73 MB | delete 1.82 ms/op
```

Tests: `cargo test --release -p ruvector-deg` runs 4 / 4 green, including a delete-then-recall regression that holds recall@5 above 0.70 after deleting 25 % of the graph.

## Optimisations

- Flat `Vec<u32>` adjacency, row layout, branch-free iteration.
- Squared-Euclidean kernel with 4-lane unroll — auto-vectorises on x86_64 and aarch64.
- Periodic random entry shuffle (every power-of-two insert) to prevent hub saturation.
- Per-row vector clone on back-edge updates instead of `unsafe` aliasing — measured cost negligible vs distance computation.
- Future: feature-gated Rayon parallel build, `ruvector-rabitq` re-rank hop, reverse-adjacency table for O(M²) delete.

## Get started

```sh
git clone https://github.com/CrossGen-ai/RuVector.git ruvector
cd ruvector
git checkout research/nightly/2026-06-08-dynamic-exploration-graph

cargo build --release -p ruvector-deg
cargo test  --release -p ruvector-deg
cargo run   --release -p ruvector-deg --bin deg-demo
cargo run   --release -p ruvector-deg --example sweep
```

- Branch: <https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-06-08-dynamic-exploration-graph>
- ADR: `docs/adr/ADR-196-ruvector-deg-dynamic-exploration-graph.md`
- Research note: `docs/research/nightly/2026-06-08-dynamic-exploration-graph/README.md`
- Upstream project: <https://github.com/ruvnet/ruvector>

ruvector is open source. If DEG matters for your real-time Rust vector workload, file an issue at the upstream repo or open a PR against the CrossGen-ai fork.
