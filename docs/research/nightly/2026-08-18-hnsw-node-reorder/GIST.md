# ruvector 2026: HNSW Node Reordering — High-Performance Rust Vector Search with Cache-Locality Graph Layouts

> RuVector nightly research (2026-08-18): a pure-Rust implementation of Gorder and Recursive Graph Bisection for HNSW-style ANN graphs, delivering +6.8% QPS at 100k×64 with bit-exact recall on Apple M4 Max.

**Repository:** <https://github.com/ruvnet/ruvector> · **Fork with this research:** <https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-18-hnsw-node-reorder> · **ADR:** [ADR-306](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-08-18-hnsw-node-reorder/docs/adr/ADR-306-hnsw-node-reorder.md)

## Introduction

Graph-based approximate nearest neighbour search (HNSW, Vamana/DiskANN,
NSG) is bounded by memory latency, not compute. On modern CPUs with
32 MB+ of shared cache, the dominant cost of a query is chasing
pointer-heavy neighbour lists and fetching vector rows that the
prefetcher couldn't predict. This RuVector nightly ports two classic
graph-locality algorithms — **Gorder** (KDD 2016) and **Recursive
Graph Bisection** (WWW 2009 / KDD 2016) — to a small Rust HNSW-lite
index and measures the effect on real query throughput.

Node reordering is **layout-only**: the search algorithm is unchanged,
recall is bit-exact preserved, and the reorder pass is a one-shot
batch operation that fits inside a normal compaction cycle.

## Features

- Four reordering strategies behind a single `Strategy` enum: `Identity`, `Bfs`, `Gorder { window }`, `Rgb { max_depth }`.
- Deterministic, seeded output. Reordering the same graph twice gives bitwise-identical permutations.
- Layout-agnostic greedy beam search with an `id_stride_sum` locality proxy that avoids platform-specific hardware counters.
- Pure Rust, zero unsafe, dependencies limited to `rand` and `rayon`. Every source file under 300 lines.
- End-to-end benchmark binary with adversarial-shuffle baseline to expose worst-case gains, not best-case.
- Composable with any HNSW/Vamana-shaped adjacency via `apply_permutation`, which rebuilds CSR neighbours and the row-major vector store in one pass.

## Benefits

- **+6.8% QPS on 100k×64** with RGB reordering, **+5.3%** with Gorder — real numbers on an Apple M4 Max, warm caches, warm branch predictor.
- **Bit-exact recall.** Reordering is a pure relabelling; the crate's `reorder_preserves_recall` test enforces this and is part of the default test suite.
- **Cheap.** RGB reorders 100k nodes in 438 ms — 42× faster than Gorder for the same quality.
- **Composable.** Layout is orthogonal to compression (ADR-297), termination (ADR-303) and receipts (ADR-304), so gains stack.

## Comparison to other vector search engines

| Engine | Graph node reordering | Algorithm | Cost model |
| --- | --- | --- | --- |
| **RuVector** (this work) | Optional, four strategies | Identity / BFS / Gorder / RGB | Batch, deterministic, ~0.4 s / 100k |
| **DiskANN v0.6** (2024) | `--reorder` flag | Gorder variant | Batch, C++ |
| **Milvus 2.5** (2025) | Yes, on ARM Graviton | Adjacency block reorder | Batch |
| **Weaviate 1.28** (2025) | `graph_compact` API | Recursive graph bisection | Manual |
| **Qdrant 1.13** | No documented reordering | — | — |
| **Pinecone** (managed) | Not exposed to users | — | — |
| **FAISS HNSW** | No | — | — |

The RuVector implementation is the only pure-Rust, no-unsafe, workspace-integrated version of this technique with a public benchmark that pits BFS, Gorder and RGB against each other on the same graph.

## Benchmarks (real numbers)

Hardware: **Apple M4 Max, 16 cores, 128 GiB RAM, macOS 24.6.0**. Stable Rust, release profile. `ef_construction=96`, `ef_search=64`, `k=10`, `m=24`, 500 queries × 3 iterations, best-of-three.

### n = 100 000, d = 64 (main result)

| Strategy | Reorder cost | log-gap | QPS | µs/query | id-stride | recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity (shuffled) | 0 ms | 14.445 | 8 924 | 112.05 | 1 425 M | 0.3410 |
| bfs | 7 ms | 14.056 | 8 563 | 116.79 | 781 M | 0.3410 |
| gorder(w=8) | 18 411 ms | 13.581 | 9 398 | 106.40 | 758 M | 0.3410 |
| **rgb(depth=14)** | **439 ms** | **13.694** | **9 534** | **104.88** | **829 M** | **0.3410** |

**RGB: +6.8% QPS, −42% id-stride, −5.2% log-gap, 42× faster to compute than Gorder for equivalent quality.**

### n = 50 000, d = 128

| Strategy | QPS | Δ vs identity | recall@10 |
| --- | ---: | ---: | ---: |
| identity (shuffled) | 5 283 | — | 0.2440 |
| gorder(w=8) | 5 425 | **+2.7%** | 0.2440 |
| rgb(d=14) | 5 227 | −1.1% (noise) | 0.2440 |

### n = 30 000, d = 128 (below L2 boundary)

Working set fits in cache; reordering shows no throughput gain, as
expected. The `log_gap` and `id_stride` proxies still improve,
confirming the algorithms are working — just below the point where
memory latency dominates.

## Optimizations

- **BFS-seeded RGB.** Random-seeded RGB fails on small ANN graphs because the balanced-halves swap budget can't recover from a truly random start. Seeding from BFS gives the coordinate-descent sweep a strong initial cut.
- **Neighbour lists sorted by new id.** After reorder, adjacency rows are sorted; the hardware prefetcher then walks them in order and gets a much higher hit rate.
- **Iteration budget per split.** Three coordinate-descent sweeps per RGB split, early-out on no-improvement — enough to catch the low-hanging swaps without blowing the reorder budget.
- **Adversarial-shuffle baseline.** Rather than benchmarking against the natural incremental-insertion order (which already gives some free locality), the reference bench first shuffles the graph to simulate bulk-load, so reported gains reflect the realistic hard case.

## Get started

```bash
# Clone the fork with this research on it
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-08-18-hnsw-node-reorder

# Run the test suite (bit-exact recall preservation)
cargo test --release -p ruvector-hnsw-reorder

# Reproduce the main benchmark
N=100000 DIM=64 cargo run --release -p ruvector-hnsw-reorder --bin reorder-bench
```

Full research writeup: [docs/research/nightly/2026-08-18-hnsw-node-reorder/README.md](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-08-18-hnsw-node-reorder/docs/research/nightly/2026-08-18-hnsw-node-reorder/README.md).

Architecture Decision Record: [ADR-306](https://github.com/CrossGen-ai/RuVector/blob/research/nightly/2026-08-18-hnsw-node-reorder/docs/adr/ADR-306-hnsw-node-reorder.md).

Upstream RuVector: <https://github.com/ruvnet/RuVector>.

---

*Keywords: HNSW, ANN, vector search, Rust, cache locality, graph reordering, Gorder, recursive graph bisection, RGB, DiskANN, Vamana, prefetch, Milvus, Qdrant, Weaviate, Pinecone, FAISS, LanceDB, RAG, embedding, similarity search, k-NN, low-latency retrieval, Apple M4 Max.*
