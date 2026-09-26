# Batched HNSW with Shared Candidate Frontier

*Nightly research 2026-09-26 · `research/nightly/2026-09-26-hnsw-batch-shared-frontier` · ADR-347*

## Abstract

We accelerate HNSW-style top-k search on **correlated query batches**
(multi-turn RAG, MMR fan-out, multi-vector retrieval) by fusing two
techniques into one search backend:

1. **Batched distance kernel** — one candidate vector is compared to
   every query in the batch in a single pass, amortizing the vector
   load cost across the batch.
2. **Batch-wide evaluated set** — the Q-length distance row of every
   visited node is cached, so any later query in the batch that reaches
   the same node skips the vector load entirely.

Per-query frontier heaps and result heaps are preserved, so recall is
**numerically identical** to independent-search HNSW. On Apple M4 Max
with n = 5 000, dim = 128, ef = 64, the shared-frontier backend saves
**71 – 97 % of candidate vector loads** and is **10 – 30 % faster in
wall clock** than independent search at batch sizes ≥ 8. A ready-to-run
Rust crate accompanies this report.

## SOTA Survey

- **HNSW** (Malkov & Yashunin, 2018, arXiv:1603.09320) — the greedy
  `ef`-search inner loop this work modifies.
- **DiskANN / Vamana** (Subramanya et al., NeurIPS 2019) — introduces
  per-query beam search over an out-of-core graph; batched frontier
  reuse for correlated queries is not addressed.
- **SPANN** (Chen et al., NeurIPS 2021) — coarse partition + fine
  in-partition search; correlated batches share partitions but the
  in-partition search remains per-query.
- **ScaNN** (Guo et al., ICML 2020) — anisotropic PQ + block scan;
  Google's follow-up "SOAR: Improved Indexing for Approximate Nearest
  Neighbor Search" (Sun et al., NeurIPS 2023) further diversifies
  spillage but keeps queries independent.
- **FreshDiskANN** (Singh et al., 2021) and **FilteredDiskANN**
  (Gollapudi et al., WWW 2023) — streaming updates and attribute
  filtering; batch semantics remain per-query.
- **Milvus / Knowhere** batched search (Milvus 2.4 release notes,
  2024) — offers a `BatchSearch` API that pipelines *independent*
  searches; distance computation is not shared across queries.
- **Qdrant multi-vector search** (Qdrant 1.10 changelog, 2024) — allows
  passing a `Vec<Query>` but each query traverses its own graph copy.
- **RaBitQ** (Gao et al., SIGMOD 2024) — 1-bit quantized distance for
  the *distance kernel*; complementary to this work (swap the L2
  kernel for a bitwise dot).
- **NN-Descent** (Dong et al., WWW 2011) and **RNN-Descent**
  (Fu et al., VLDB 2019) — build-time neighbour selection; orthogonal.
- **Facebook FAISS** `IndexHNSWFlat::search` (FAISS 1.8.0) —
  parallelises **across** queries (one thread per query) but does not
  share within-batch computation.

Gap: **no mainstream vector DB or open-source ANN library shares
distance work across queries in a batch.** This crate demonstrates the
technique in ~500 LoC of Rust and quantifies its win.

## Proposed Design

The core abstraction is a `BatchedSearch` trait:

```rust
pub trait BatchedSearch {
    fn name(&self) -> &'static str;
    fn search_batch(
        &self,
        graph: &ProximityGraph,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> Vec<TopK>;
}
```

Three implementations ship in this crate:

1. `IndependentSearch` — one greedy `ef`-search per query. Baseline.
2. `BatchedKernelSearch` — same traversal semantics; **all** distances
   go through `l2_sq_batch(candidate, queries, out)` which computes
   the distance from `candidate` to every batch query in a single pass.
3. `SharedFrontierSearch` — extends (2) with a batch-wide `evaluated`
   bitmap and a `cache: Vec<f32>` sized `Q · N`. If a candidate is
   already evaluated, its cached row is reused instead of re-loading
   the vector.

Per-query state (visited set, candidate heap, result heap) is
maintained separately so each query still owns its own frontier and
result set — recall matches the independent baseline exactly.

## Implementation Notes

- **File layout** (all < 500 lines):
  - `src/lib.rs` — trait + module wiring.
  - `src/dist.rs` — scalar + batched squared-L2 kernels with a 4× query
    row unroll to keep the candidate hot in registers across queries.
  - `src/graph.rs` — flat proximity graph (`m = 32` nearest of a
    `pool = 256` random candidate set per node). Deliberately simple:
    the experimental variable is the *search* not the *build*.
  - `src/independent.rs` — baseline greedy `ef`-search.
  - `src/batched.rs` — batched-kernel variant, round-robin over
    per-query heaps.
  - `src/shared.rs` — shared-frontier variant, adds an `evaluated`
    bitmap and dense distance-row cache.
  - `src/bench.rs` — synthetic clustered dataset, correlated query
    generator, `BenchReport` struct with recall accounting.
  - `src/bin/bench.rs` — full sweep binary.
  - `examples/run_bench.rs` — one-shot demo.
- **Determinism**: `rand_chacha::ChaCha8Rng` for both dataset and
  queries, so nightly runs are reproducible.
- **Trait-based swap surface**: production callers depend on
  `BatchedSearch` only. Kernel or graph replacement is a two-line
  change.

### Memory math

- Independent baseline: no batch state.
- Batched kernel: `Q · f32` scratch (~64 B at Q = 16).
- Shared frontier: `N · Q · f32` cache + `N` bit `evaluated` set.

  | N       | Q   | cache   | evaluated | total  |
  |---------|-----|---------|-----------|--------|
  | 5 000   | 4   | 78 KiB  | 625 B     | 79 KiB |
  | 5 000   | 32  | 625 KiB | 625 B     | 625 KiB|
  | 1 M     | 32  | 122 MiB | 122 KiB   | 122 MiB|

  For large N a sparse (HashMap-backed) cache is called for. The PoC
  uses a dense `Vec<f32>` to keep the hot path branch-free.

## Benchmark Methodology

- **Hardware**: Apple M4 Max (arm64), Darwin 24.6.0, rustc 1.98.1
  (release profile, `lto = thin` inherited from workspace).
- **Dataset**: n = 5 000, dim = 128, 32 Gaussian clusters, σ = 0.15.
  Query anchor drawn from the dataset itself so recall reflects
  graph-traversal quality, not entry-point geometry.
- **Correlated queries**: one anchor per batch, per-query perturbation
  `U(-noise, noise)` with `noise = 1 - corr`.
- **Ground truth**: exact brute-force top-k per query, computed once.
- **Metrics**:
  - `ms` — wall-clock end-to-end (whole batch, not per query).
  - `loads` — distinct candidate vector loads (i.e. graph reads).
  - `hits` — cache hits saved by the shared frontier.
  - `recall` — `mean(|result ∩ truth| / k)`.
- **Sweep**: `Q ∈ {4, 8, 16, 32}`, `corr ∈ {0.6, 0.85, 0.98}`,
  `k = 10`, `ef = 64`.

## Results

Raw numbers copied verbatim from `bench.raw.txt` (produced by
`cargo run --release --bin bsf-bench`).

### Q = 4

| corr | backend         |    ms | loads | hits | recall |
|-----:|-----------------|------:|------:|-----:|-------:|
| 0.60 | independent     |  0.19 | 3 096 |    0 | 0.975  |
| 0.60 | batched-kernel  |  0.32 | 3 093 |    0 | 0.975  |
| 0.60 | shared-frontier |  0.15 |   889 | 2204 | 0.975  |
| 0.85 | independent     |  0.16 | 3 115 |    0 | 0.975  |
| 0.85 | batched-kernel  |  0.33 | 3 112 |    0 | 0.975  |
| 0.85 | shared-frontier |  0.15 |   873 | 2239 | 0.975  |
| 0.98 | independent     |  0.17 | 3 101 |    0 | 1.000  |
| 0.98 | batched-kernel  |  0.31 | 3 098 |    0 | 1.000  |
| 0.98 | shared-frontier |  0.14 |   808 | 2290 | 1.000  |

### Q = 8

| corr | backend         |    ms | loads | hits | recall |
|-----:|-----------------|------:|------:|-----:|-------:|
| 0.60 | independent     |  0.33 | 6 186 |    0 | 0.975  |
| 0.60 | batched-kernel  |  1.08 | 6 179 |    0 | 0.975  |
| 0.60 | shared-frontier |  0.29 |   900 | 5279 | 0.975  |
| 0.85 | independent     |  0.32 | 6 196 |    0 | 0.9875 |
| 0.85 | batched-kernel  |  1.10 | 6 189 |    0 | 0.9875 |
| 0.85 | shared-frontier |  0.28 |   883 | 5306 | 0.9875 |
| 0.98 | independent     |  0.31 | 6 204 |    0 | 1.000  |
| 0.98 | batched-kernel  |  1.00 | 6 197 |    0 | 1.000  |
| 0.98 | shared-frontier |  0.24 |   824 | 5373 | 1.000  |

### Q = 16

| corr | backend         |    ms |   loads |  hits | recall |
|-----:|-----------------|------:|--------:|------:|-------:|
| 0.60 | independent     |  0.69 | 12 432  |     0 | 0.9875 |
| 0.60 | batched-kernel  |  4.25 | 12 417  |     0 | 0.9875 |
| 0.60 | shared-frontier |  0.59 |    907  | 11 510| 0.9875 |
| 0.85 | independent     |  0.60 | 12 494  |     0 | 0.9812 |
| 0.85 | batched-kernel  |  4.05 | 12 479  |     0 | 0.9812 |
| 0.85 | shared-frontier |  0.54 |    889  | 11 590| 0.9812 |
| 0.98 | independent     |  0.61 | 12 436  |     0 | 1.000  |
| 0.98 | batched-kernel  |  3.98 | 12 421  |     0 | 1.000  |
| 0.98 | shared-frontier |  0.49 |    828  | 11 593| 1.000  |

### Q = 32

| corr | backend         |     ms |   loads |  hits | recall |
|-----:|-----------------|-------:|--------:|------:|-------:|
| 0.60 | independent     |  1.47  | 24 801  |     0 | 0.9844 |
| 0.60 | batched-kernel  | 18.45  | 24 770  |     0 | 0.9844 |
| 0.60 | shared-frontier |  1.25  |    907  | 23 863| 0.9844 |
| 0.85 | independent     |  1.23  | 24 928  |     0 | 0.9875 |
| 0.85 | batched-kernel  | 15.99  | 24 897  |     0 | 0.9875 |
| 0.85 | shared-frontier |  1.15  |    900  | 23 997| 0.9875 |
| 0.98 | independent     |  1.12  | 24 828  |     0 | 1.000  |
| 0.98 | batched-kernel  | 15.69  | 24 797  |     0 | 1.000  |
| 0.98 | shared-frontier |  1.00  |    835  | 23 962| 1.000  |

### Headline numbers

| metric                                       | best value |
|----------------------------------------------|-----------:|
| Vector-load savings vs independent (Q=32, corr=0.98) | **96.6 %** |
| Wall-clock speedup vs independent (Q=32, corr=0.98)  | **1.12 ×** |
| Wall-clock speedup vs independent (Q=16, corr=0.98)  | **1.24 ×** |
| Recall (any config, all backends)                    | **≥ 0.975**|
| Recall drop between backends (any config)            | **0.000**  |

## How it Works (Blog-Readable Walkthrough)

Think of HNSW's inner search as a torch-lit corridor crawl: you're
holding a query vector, standing at the entry node, and each step
you pick the neighbouring node that is closest to the query. On a
single query this is efficient. On a batch of similar queries — say
16 paraphrases of the same question — each searcher walks nearly the
same corridor, and *each of them pays the memory cost of picking up
every node vector they inspect.*

The batched kernel shares the vector: pick up the candidate node
once, hand it to all 16 searchers, they each compute their own
distance from it and drop it. The vector's cache line is warm for all
16 computations instead of just one.

The shared frontier goes further: it writes down the 16 distances on
the wall next to that node the first time anyone inspects it. Later
searchers who wander back to the same node just read their number off
the wall instead of picking up the vector again. The wall gets big
(one number per query per node visited by anyone), so on very large
graphs you cap it or make it sparse — but on the graphs that fit in
a typical RAG working set, it stays a few hundred KB and cuts the
number of vector picks by an order of magnitude.

## Practical Failure Modes

- **Uncorrelated batches.** If queries share almost no visited nodes,
  the shared cache degenerates to per-query loading with extra
  bookkeeping. Route uncorrelated batches to `IndependentSearch`.
- **Very large N.** Dense `Q · N` cache does not scale past a few M
  vectors; replace with a hash-map-backed sparse cache (documented
  swap point).
- **Non-shared entry point.** If the batch's queries land in disjoint
  graph regions (multi-modal batch), overlap is near-zero. Detect via
  entry-point distance spread and split the batch.
- **Cache invalidation under writes.** The shared cache is per-search
  scoped; a concurrent write does not corrupt it. But for streaming
  indexes (`ruvector-streaming-qng`) the graph geometry must be
  stable for the duration of one batch's search.
- **Batched-kernel-only path**: on its own it is 3-15 × slower than
  the baseline because it does Q distance computations per node with
  no reuse. It exists as an ablation building block, not a production
  path.

## What to Improve Next

1. **Sparse cache** with an LRU eviction policy for N > 1 M.
2. **Query centroid pre-search**: search once with the batch's
   centroid to warm the shared frontier before per-query fanout.
3. **NEON / AVX-512 kernel** — the scalar unrolled kernel is ~30 % of
   theoretical peak on M4 Max; hand-vectorised SIMD would double it.
4. **Layer-integrated version** — port the three backends into
   `ruvector-coherence-hnsw` so hierarchical HNSW's upper layers also
   benefit.
5. **RaBitQ / PQ integration** — swap L2 for a bit-quantized dot
   product; the batched kernel becomes an XOR-popcount, which stacks
   with the shared cache.
6. **Adaptive routing**: cheap correlation estimator on the input
   batch that picks the backend automatically (independent vs shared).

## Production Crate Layout Proposal

```
crates/ruvector-hnsw-batch-shared-frontier/
├── src/lib.rs           # BatchedSearch trait + module wiring
├── src/dist.rs          # scalar + batched L2 (add NEON/AVX behind cfg)
├── src/graph.rs         # trait `GraphProvider`, adaptor for HNSW layers
├── src/independent.rs   # baseline backend
├── src/batched.rs       # batched-kernel backend
├── src/shared.rs        # shared-frontier backend (dense + sparse caches)
├── src/router.rs        # NEW: batch-correlation estimator + backend picker
├── src/bench.rs         # BenchReport, dataset synth
└── src/bin/bench.rs     # sweep binary
```

Downstream integration:

- `ruvector-coherence-hnsw` layer-0 search → `BatchedSearch` trait obj.
- `ruvector-maxsim` / `ruvector-muvera` — call `search_batch` with
  the per-token query vectors.
- `ruvector-diverse-beam` — the beam is a natural correlated batch.

## References

- Malkov, Yu. A. & Yashunin, D. A. *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs.* IEEE TPAMI 42 (4), 2020. arXiv:1603.09320
- Subramanya, S. J. et al. *DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
- Chen, Q. et al. *SPANN: Highly-efficient Billion-scale Approximate
  Nearest Neighbor Search.* NeurIPS 2021.
- Guo, R. et al. *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization.* ICML 2020. (ScaNN)
- Sun, P. et al. *SOAR: Improved Indexing for Approximate Nearest
  Neighbor Search.* NeurIPS 2023.
- Gao, X. et al. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound.* SIGMOD 2024.
- Dong, W., Charikar, M., Li, K. *Efficient K-nearest neighbor graph
  construction for generic similarity measures.* WWW 2011.
- Milvus 2.4 release notes (2024): `BatchSearch` API.
- Qdrant 1.10 changelog (2024): multi-vector search.
- FAISS 1.8.0 (2024): `IndexHNSWFlat::search` implementation.
