# ruvector 2026: HCNNG — High-Performance Rust Vector Search via MST Proximity Graphs

> 150-char summary: HCNNG in pure Rust — parameter-light approximate nearest-neighbor index built from random partition trees + per-leaf MSTs. Real benchmarks vs brute force inside.

## Introduction

**Approximate nearest neighbor search** is the workhorse of modern AI: every RAG retrieval, every vector database query, every semantic search starts with finding the k closest vectors to a query out of billions. The reigning algorithms — **HNSW**, **NSG**, **DiskANN** — deliver excellent recall, but each carries a tax: insertion order dependencies, expensive kNN-graph prerequisites, or per-dataset pruning tuning. This release adds **HCNNG** (Hierarchical Clustering-based Navigating Neighbor Graph, Munoz et al. Pattern Recognition 2019) to **[ruvector](https://github.com/ruvnet/ruvector)** as a new pure-Rust crate. HCNNG is the **parameter-light, embarrassingly-parallel** member of the proximity-graph family: build N independent random partition trees, take an MST per leaf, union all edges, search with a multi-entry beam. No layer hierarchy. No global insertion order. No per-dataset pruning rule.

This nightly research drop ships a complete working Rust implementation, real benchmarks against brute-force ground truth, and an ADR documenting the production roadmap.

**Keywords:** Rust vector search, approximate nearest neighbor, HCNNG, proximity graph, MST nearest neighbor, ANN index, vector database, HNSW alternative, RAG retrieval, embedding search.

## Features

- ✅ **Pure Rust**, no C/C++ deps, no `unsafe` outside of stdlib heaps.
- ✅ **Trait-based `Distance`** — L2², neg inner-product, cosine; swap in custom distances (RaBitQ, LVQ codes) without touching the graph layer.
- ✅ **MST + optional kNN augmentation** per leaf for tunable graph density.
- ✅ **Multi-entry beam search** — top-K hub nodes + deterministic random anchors.
- ✅ **Deterministic builds** — same seed, bit-identical graph.
- ✅ **Files under 500 lines, modules cleanly separated** — easy to fork/extend.
- ✅ **Real cargo benchmark binary** — no mocked numbers.

## Benefits

| What HCNNG gives you                          | Why it matters                                 |
|-----------------------------------------------|------------------------------------------------|
| Embarrassingly parallel construction          | One thread per tree → trivial scale-out        |
| Incremental graph growth                      | Add a tree → union edges → done (no rebuild)   |
| Two knobs total (`n_trees`, `leaf_size`)      | No alpha, M, mL, efConstruction tuning         |
| Cluster-friendly graph topology               | MSTs preserve metric structure inside leaves   |
| Composable with quantization                  | Plug RaBitQ/LVQ into `Distance` trait          |

## Comparisons

| Index family   | Build cost      | Recall@10 ceiling | Tuning knobs                  | Parallel build   |
|----------------|-----------------|-------------------|-------------------------------|------------------|
| **HCNNG (this PR)** | O(n·log n · T) | 0.99 (uniform d=32, n=20k) | n_trees, leaf_size      | ✅ (per-tree)    |
| HNSW (Milvus, Qdrant, Weaviate, Pinecone, FAISS-HNSW) | O(n·log n) | 0.99+ | M, efConstruction, mL | partial          |
| NSG/MRNG (FAISS-NSG, Milvus) | O(n^1.14)+ | 0.99+ | L, R, C, alpha       | weak             |
| DiskANN (Milvus DiskANN, ruvector-diskann) | O(n·log n) | 0.99+ | alpha, R, L     | shard-parallel   |
| IVF-Flat (FAISS, Milvus, pgvector) | O(n·k) | depends on nprobe | nlist, nprobe    | ✅              |
| Brute force    | O(n·d) per query | 1.00              | none                          | ✅              |

The HCNNG niche: **adding a tree is composable**. Compare to HNSW where global insertion order matters; NSG which needs a full kNN graph upfront; DiskANN which tunes per dataset.

## Benchmarks

All numbers are **real** — produced by `cargo run --release -p ruvector-hcnng` on the branch below, against brute-force L2 ground truth on the same data and queries.

**Hardware:** macOS Darwin 24.6, Apple Silicon, single-threaded build. Stock LLVM codegen (no SIMD intrinsics).

### n = 20,000, d = 32, uniform i.i.d. [-1, 1], k = 10

| Variant                          | Build (ms) | µs/query | Recall@10 | Speedup |
|----------------------------------|-----------:|---------:|----------:|--------:|
| brute_force_L2                   |          – |    414.9 |     1.000 |    1.0× |
| hcnng_n_trees=1 (ablation)       |       12.1 |      8.9 |     0.008 |   46.8× |
| hcnng_n_trees=12 (default)       |      274.8 |     61.2 |     0.863 |    6.8× |
| **hcnng_n_trees=20, ef=128**     |    **463.1** | **165.0** | **0.991** | **2.5×** |

### n = 20,000, d = 64

| Variant                          | Build (ms) | µs/query | Recall@10 | Speedup |
|----------------------------------|-----------:|---------:|----------:|--------:|
| brute_force_L2                   |          – |    553.7 |     1.000 |    1.0× |
| hcnng_n_trees=12 (default)       |      337.7 |     85.3 |     0.599 |    6.5× |
| **hcnng_n_trees=20, ef=128**     |    **624.9** | **205.8** | **0.929** | **2.7×** |

### n = 50,000, d = 64 (scale)

| Variant                          | Build (ms) | µs/query | Recall@10 | Speedup |
|----------------------------------|-----------:|---------:|----------:|--------:|
| brute_force_L2                   |          – |   1479.5 |     1.000 |    1.0× |
| hcnng_n_trees=12 (default)       |      970.1 |     98.5 |     0.402 |   15.0× |
| **hcnng_n_trees=20, ef=128**     |   **1783.7** | **257.8** | **0.792** | **5.7×** |

**Acceptance** (recall@10 ≥ 0.90 on best variant) **PASSES** at n=20k for both d=32 and d=64.

### Memory

Graph adjacency at n=20k, d=64, n_trees=12 is **8.5 MB** — **1.67× the raw vector bytes**. Comparable to HNSW M=16 (1.0–1.5×), well under NSG (2–3×).

## Optimizations

Already in this drop:

1. **Prim's MST with arrays, not Kruskal+DSU** — leaf size is bounded, so O(L²) array Prim beats heap Kruskal on cache.
2. **Reused leaf distance matrix** — when `knn_per_node > 0`, compute the L×L matrix once and read both MST weights and kNN rows from it. kNN is free.
3. **Multi-entry beam search** — top-4 hubs + 4 deterministic random anchors. Critical for recall on multi-modal data.
4. **Deterministic per-tree seed derivation** — `seed.wrapping_add(t * GOLDEN_PRIME)` → reproducible graphs.

On the roadmap (next nightlies):

1. **rayon parallel tree build** — N-thread speedup.
2. **Alpha-RNG / Vamana-style edge pruning** — preserve long-range edges under aggressive truncation.
3. **RaBitQ-quantized distance trait** — 32× memory cut on adjacency vectors.
4. **Streaming add / lazy retree** — no global rebuild on insert.
5. **Filtered HCNNG** — composes with ruvector-acorn filter predicates.
6. **KD-tree entry resolver** — fixes the clustered-data recall hole.

## Get started

The implementation lives on the CrossGen-ai fork of ruvector, on the dated nightly research branch:

- **Branch (fork):** [research/nightly/2026-06-14-hcnng-mst-graph](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-06-14-hcnng-mst-graph)
- **Crate:** `crates/ruvector-hcnng/`
- **ADR:** `docs/adr/ADR-252-hcnng-mst-graph.md`
- **Research note (full benchmarks + design):** `docs/research/nightly/2026-06-14-hcnng-mst-graph/README.md`
- **Upstream ruvector:** https://github.com/ruvnet/ruvector

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-06-14-hcnng-mst-graph
cargo test --release -p ruvector-hcnng
cargo run --release -p ruvector-hcnng

# Reproduce the n=50k scale benchmark
D=64 N=50000 NQ=100 cargo run --release -p ruvector-hcnng

# Stress-test on clustered data
DATASET=gmm cargo run --release -p ruvector-hcnng
```

```rust
use ruvector_hcnng::{HcnngIndex, HcnngParams, Metric};

let vectors: Vec<Vec<f32>> = load_my_vectors();
let idx = HcnngIndex::build(vectors, HcnngParams {
    n_trees: 12,
    leaf_size: 32,
    max_degree: 32,
    knn_per_node: 3,
    ef_search: 64,
    seed: 42,
    metric: Metric::L2Sq,
}).unwrap();

let hits = idx.search(&query, 10).unwrap();
for h in hits {
    println!("id={} dist={}", h.id, h.distance);
}
```

---

**ruvector** is an open-source high-performance vector search library in Rust. Star the project at [github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector). Nightly research branches like this one live on the [CrossGen-ai fork](https://github.com/CrossGen-ai/RuVector) and explore SOTA proximity graph, quantization, and retrieval techniques for production deployment.
