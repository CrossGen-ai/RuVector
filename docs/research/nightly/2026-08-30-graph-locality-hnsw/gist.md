# ruvector 2026: Graph-Locality Reordering — High-Performance Rust Vector Search HNSW ANN

**Free +19 % QPS on HNSW nearest-neighbor search with bit-identical recall — pure Rust, zero unsafe, one physical-storage permutation.**

## Introduction

HNSW is the reigning graph-based Approximate Nearest Neighbor (ANN) index —
it powers Milvus, Qdrant, Weaviate, Pinecone, LanceDB, and FAISS-HNSW. At
high recall its cost is not distance arithmetic; it is *pointer-chasing*:
every visited node dereferences its neighbor list into a flat vector buffer,
and every neighbor is a cache miss when physical order is uncorrelated with
graph order (the universal default: insertion order).

**ruvector-graph-locality-hnsw** — the 2026-08-30 nightly experiment
inside [ruvector](https://github.com/ruvnet/RuVector) — reorders that flat
buffer so graph-adjacent vectors are memory-adjacent. Measured lift: **+17-19 %
QPS on 50k×128 HNSW at bit-identical recall and bit-identical work counts**.

Rust ANN, Rust vector search, Rust HNSW, Rust cache-locality, Rust vector
database, high-performance ANN, low-latency retrieval, RAG infrastructure.

## Features

- `ReorderStrategy` trait with three landing strategies: **Identity** (baseline),
  **BFS from entry point**, **Reverse Cuthill-McKee** over symmetrized L0.
- Pure function of the source index — pass is idempotent, correctness-preserving,
  and produces a `new_to_old: Vec<u32>` inverse map for external id references.
- `mean_edge_gap()` — a hardware-independent locality metric usable as a CI
  regression signal.
- Full instrumented HNSW query in ~120 lines: exposes `nodes_visited`,
  `distance_calls`, top-k, ef-search — enough to prove the QPS delta comes
  from cache traffic, not from doing less work.
- 5 files total, `rand` + `rand_distr` the only non-std deps, `#![forbid(unsafe_code)]`.

## Benefits

- **Bit-identical recall.** Not "close to" — reordering permutes memory, not
  the graph or the algorithm. Regression tests assert set-equality on
  returned ids.
- **Amortized cost.** BFS reorder on 50k×128: 12 ms. RCM: 60 ms. Both
  disappear against query throughput after ~1 query.
- **Composes.** Orthogonal to quantization (RaBitQ, PQ, 4-bit), filtering
  (ACORN), and coherence-aware search. Layer it on and take the lift.
- **Snapshot-friendly.** Reorder is the natural time to compact tombstones
  and rewrite snapshots with a `layout_hash` gate.

## Comparisons

| Engine       | Vector layout after graph build | Reorder pass available |
|--------------|---------------------------------|-----------------------:|
| hnswlib      | insertion order                 | no                     |
| FAISS-HNSW   | insertion order                 | no                     |
| Milvus       | insertion order                 | no                     |
| Qdrant       | insertion order                 | no                     |
| Weaviate     | insertion order                 | no                     |
| Pinecone     | opaque, no public reorder pass  | no                     |
| LanceDB      | IVF posting-list reorder, no HNSW-graph reorder | partial |
| **ruvector-graph-locality-hnsw** | **pluggable (Identity / BFS / RCM / …)** | **yes** |

## Benchmarks (REAL numbers)

Synthetic 128-dim Gaussian mixture, 64 clusters (σ=0.5). Reproducer command
in the crate; every number below is emitted by `target/release/reorder-bench`.
Hardware: M-series Mac laptop, single thread, `cargo build --release`, LTO off.

**n = 50 000, 500 queries × 3 reps, k = 10:**

| ef  | strategy | edge gap | QPS       | recall@10 | avg visited |
|-----|----------|---------:|----------:|----------:|------------:|
| 30  | identity |  12 226  |    32 130 |     0.244 |       427.1 |
| 30  | bfs      |   8 536  |**38 103** |     0.244 |       427.1 |
| 30  | rcm      |   2 425  |    37 307 |     0.244 |       427.1 |
| 60  | identity |  12 226  |    21 913 |     0.280 |       545.2 |
| 60  | bfs      |   8 536  |**25 927** |     0.280 |       545.2 |
| 60  | rcm      |   2 425  |    24 719 |     0.280 |       545.2 |
| 120 | identity |  12 226  |    16 086 |     0.302 |       634.3 |
| 120 | bfs      |   8 536  |**18 927** |     0.302 |       634.3 |
| 120 | rcm      |   2 425  |    18 222 |     0.302 |       634.3 |

**BFS +17.6 % to +19.0 % QPS. RCM +13.2 % to +16.1 %. Recall identical.**

**n = 20 000, ef = 60:** identity 32 509 QPS → BFS 33 948 → **RCM 40 542 QPS
(+25 %)**. At smaller scale RCM occasionally overtakes BFS.

## Optimizations

- **BFS from entry point** — dense pack the hot start basin every query
  touches. Wins at large ef and large n.
- **Reverse Cuthill-McKee** — minimize global mean edge gap. Wins on
  hub-flat workloads and smaller graphs.
- **Trait-plug** — Louvain, METIS, node2vec-sort, and learned layouts all
  slot in without touching the query path.
- **`mean_edge_gap` regression guard** — a hardware-independent proxy your
  CI can pin without needing `perf_event_open`.

## Get started

Nightly-research fork branch (NOT an upstream PR):

- **Branch:** [github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-30-graph-locality-hnsw](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-30-graph-locality-hnsw)
- **Upstream project:** [github.com/ruvnet/ruvector](https://github.com/ruvnet/ruvector)
- **Crate:** `crates/ruvector-graph-locality-hnsw`
- **ADR:** ADR-341
- **Research write-up:** `docs/research/nightly/2026-08-30-graph-locality-hnsw/README.md`

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-08-30-graph-locality-hnsw
cargo test  --release -p ruvector-graph-locality-hnsw
cargo build --release -p ruvector-graph-locality-hnsw
N=50000 D=128 Q=500 EF=30,60,120 \
  ./target/release/reorder-bench
```
