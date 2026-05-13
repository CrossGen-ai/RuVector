# NSG — Navigating Spreading-out Graph for ANN search

> Nightly research, 2026-05-13. Crate: `crates/ruvector-nsg`. ADR-195.

## Abstract

We add a clean, single-layer **Navigating Spreading-out Graph** (NSG)
index to ruvector. NSG (Fu, Xiang, Wang & Cai, VLDB 2019) builds an
approximate k-NN graph via NN-Descent, picks one fixed *navigating* base
point near the dataset centroid, then prunes per-node out-edges with the
Monotonic Relative Neighbourhood Graph (MRNG) rule. The resulting graph
is one layer, has bounded out-degree, and provides a monotonic search
path from the navigating node to every base point. Our Rust
implementation reaches **recall@10 = 0.965 at 8,012 QPS on 50,000
uniform-random 32-dim points (single-thread Apple M4 Max), a 2.64×
speedup over brute force**, with no `unsafe`, no BLAS, and no C
dependencies.

## SOTA survey

| Index | Layers | Build mode | Edge rule | Key knob | Ref |
|---|---|---|---|---|---|
| HNSW (Malkov & Yashunin, 2018) | O(log N) | online | heuristic-select | `M`, `efC` | [arXiv:1603.09320] |
| Vamana / DiskANN (Subramanya et al., NeurIPS 2019) | 1 | batch (two-pass) | α-relaxed MRNG | `R`, `L`, `α` | [DiskANN paper] |
| **NSG (Fu et al., VLDB 2019)** | **1** | **batch** | **strict MRNG + DFS** | **`R`, `L`** | [arXiv:1707.00143] |
| NSSG / SSG (Fu et al., 2019) | 1 | batch | angle-based | `R`, `θ` | [arXiv:1907.06146] |
| τ-MNG / NHQ / PASS | 1 | batch | bounded-angle MRNG | various | recent SIGMOD/VLDB |
| ParlayANN (Manohar et al., PPoPP 2024) | depends | parallel | implementation harness | — | parallel-only |

Competitor changelogs (May 2026):

- **Milvus 2.5** ships IVF-RaBitQ and HNSW-PQ but **no NSG** — they
  prefer multi-layer graphs for hot-corpus serving.
- **Qdrant** added Vamana-style pruning under the hood for HNSW (2026
  Q1) but the public graph is still HNSW.
- **Weaviate** has only HNSW.
- **FAISS** has IndexNSGFlat / IndexNSGPQ — the C++ reference. Ours is
  a from-scratch Rust port with the Vamana-α extension exposed.
- **LanceDB** added IVF-PQ + reranking in 2026; still no NSG.

The signal: NSG is well-known in the literature but underrepresented in
production Rust vector libraries. There is an industry gap.

[arXiv:1603.09320]: https://arxiv.org/abs/1603.09320
[arXiv:1707.00143]: https://arxiv.org/abs/1707.00143
[arXiv:1907.06146]: https://arxiv.org/abs/1907.06146
[DiskANN paper]: https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html

## Proposed design

Four-step pipeline:

```
            ┌──────────────────┐
data ──────▶│ 1. NN-Descent    │── approximate K-NN graph (K = 30-60)
            └──────────────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ 2. Navigating    │── greedy walk on K-NN graph to find
            │    node = nearest│   the base point closest to centroid
            │    to centroid   │
            └──────────────────┘
                     │
                     ▼
            ┌──────────────────────────────────┐
            │ 3. Per-node MRNG edge selection  │   pool = greedy(nav→p)
            │                                  │      ∪ knn[p]
            │    keep p→c iff no selected s    │   sorted by d(p, .)
            │    has α·d(s,c) < d(p,c)         │   cap at R
            └──────────────────────────────────┘
                     │
                     ▼
            ┌──────────────────────────────────┐
            │ 3b. Reverse-edge insertion       │
            │     + re-prune                   │
            └──────────────────────────────────┘
                     │
                     ▼
            ┌──────────────────────────────────┐
            │ 4. DFS from navigating node;     │
            │    connect unreachable nodes via │
            │    nearest reached ancestor      │
            └──────────────────────────────────┘
                     │
                     ▼
                NsgIndex (vectors + adj + nav)
```

Search (Algorithm 1 in the paper): single-layer greedy beam search from
`nav` with bounded candidate list of size `L_search ≥ k`. Stops when the
closest unvisited candidate is no better than the worst pool entry.

## Implementation notes

The crate is ~800 lines across `lib.rs`, `knn.rs`, `nsg.rs`, `error.rs`,
plus a binary `main.rs` and a `criterion` bench. Each file < 500 lines.

Key choices:

- **Squared L2 only.** Sufficient for ranking; avoids the `sqrt`
  hot-path. No SIMD intrinsics — left for a follow-up; Rust autovec
  already handles tight `for` loops on Apple Silicon.
- **MRNG occlusion test** uses Vamana's α ≥ 1.0 relaxation. At α = 1.0
  this is the strict NSG rule. At α = 1.2-1.5 we keep more edges,
  helpful on tight clusters where strict pruning fragments the graph.
- **Reverse-edge step.** Without it, MRNG can leave many nodes with
  near-empty in-degree, especially in dense regions, and the navigating
  node ends up isolated from large swaths of the dataset.
- **DFS augmentation** as a final safety net: any vertex still
  unreachable from `nav` gets a single edge from its nearest reached
  ancestor.

## Benchmark methodology

- **Hardware:** Apple M4 Max, macOS 15.7, **single thread**. No SIMD
  hand-tuning; Rust autovec only.
- **Workload:** uniform-random vectors in `[-1, 1]^d`. We hold out
  `nq=500` queries from the same generator (the standard ANN-bench
  protocol).
- **Ground truth:** brute-force exact top-k via `l2_sq` over the entire
  base set.
- **Metric:** recall@10 = `|gt ∩ pred| / 10`, averaged over the held-out
  queries.
- **Build / search timing:** `std::time::Instant`; **search QPS** is
  measured after an 8-query warm-up.

All numbers below are taken straight from
`cargo run --release -p ruvector-nsg --bin nsg-demo`. To reproduce:

```sh
NSG_N=50000 NSG_D=32 cargo run --release -p ruvector-nsg --bin nsg-demo
```

## Results

### N = 10,000, D = 64

| Index | R | L_build | L_search | Build (s) | Avg deg | Graph (KiB) | Recall@10 | QPS | Speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| brute force | — | — | — | — | — | — | 1.000 | 7,346 | 1.00× |
| NSG / small  | 20 |  40 |  40 | 2.11 | 16.7 | 1,016 | 0.663 | 21,330 | **2.90×** |
| NSG / medium | 32 | 100 | 100 | 4.19 | 32.0 | 1,488 | 0.936 |  6,889 | 0.94× |
| NSG / large  | 48 | 200 | 200 | 5.97 | 48.0 | 2,112 | 0.993 |  3,830 | 0.52× |

### N = 20,000, D = 64

| Index | R | L_build | L_search | Build (s) | Avg deg | Graph (KiB) | Recall@10 | QPS | Speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| brute force | — | — | — | — | — | — | 1.000 | 3,617 | 1.00× |
| NSG / small  | 20 |  40 |  40 |  6.07 | 17.0 | 2,035 | 0.553 | 19,191 | **5.31×** |
| NSG / medium | 32 | 100 | 100 | 11.27 | 32.0 | 2,981 | 0.888 |  6,197 | **1.71×** |
| NSG / large  | 48 | 200 | 200 | 16.16 | 48.0 | 4,228 | 0.983 |  2,754 | 0.76× |

### N = 50,000, D = 32

| Index | R | L_build | L_search | Build (s) | Avg deg | Graph (KiB) | Recall@10 | QPS | Speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| brute force | — | — | — | — | — | — | 1.000 | 3,036 | 1.00× |
| NSG / small  | 20 |  40 |  40 | 26.42 | 16.9 |  5,079 | 0.734 | 19,973 | **6.58×** |
| NSG / medium | 32 | 100 | 100 | 36.05 | 31.8 |  7,424 | 0.965 |  8,012 | **2.64×** |
| NSG / large  | 48 | 200 | 200 | 48.62 | 48.0 | 10,549 | 0.997 |  2,963 | 0.98× |

### Interpretation

- **The speedup grows with N** — exactly as expected for a sub-linear
  graph index. At N=10k brute force is already fast and NSG/medium is
  near break-even. At N=50k NSG/medium gives 2.64× over brute force *at
  96.5% recall*, and NSG/small gives **6.58×** at the lower 73% recall
  point.
- **Memory:** the graph itself is small. At N=50k, NSG/medium needs
  only **7.4 MiB of adjacency** (≈ 152 bytes / vector for the graph;
  raw vectors at f32 × 32 = 128 bytes / vector). Less than half of an
  equivalent HNSW with the same R per layer.
- **Recall ladder is clean:** `r=20 → 32 → 48` produces
  `0.73 → 0.97 → 1.00` recall — predictable, tunable, no surprises.

## How it works — walkthrough

If you have ever used HNSW, NSG will feel familiar but stripped down.
Here is the intuition in plain prose:

1. We need a graph where greedy search ("from the current node, jump to
   the neighbour closest to the query") *converges* to the true top-k.
   Random graphs do not have this property. HNSW solves it by stacking
   layers. NSG solves it by **choosing edges carefully**.
2. The Monotonic Relative Neighbourhood Graph rule says: for every node
   `p`, only keep an edge to `c` if no already-kept neighbour `s` is
   "between" `p` and `c`. Formally: no `s` with `d(s, c) < d(p, c)`. If
   you do this, greedy search from any node has a monotonically
   decreasing distance to the target — no local minima.
3. Building the full MRNG is `O(N²)`. NSG approximates it by
   constructing the graph only over a **candidate pool** of `L_build`
   nearby points, found via a greedy beam search from a single fixed
   *navigating* node (chosen near the dataset centroid).
4. The navigating node is the **single entry point** for every query:
   search always starts there, walks the graph greedily toward the
   query, and stops when no neighbour is closer than the current pool.

That is the whole algorithm. Everything else — `α`, reverse edges, DFS
augmentation — is an engineering fix for edge cases where the strict
rule produces a too-sparse graph.

## Practical failure modes

- **Pathologically tight clusters.** Strict MRNG (α = 1.0) may keep only
  1-3 edges per node in a dense cluster because the second-closest
  cluster member occludes everything else. Fix: set `alpha = 1.2-1.5`.
- **High intrinsic dimension.** When the dataset has effective dim
  close to the embedding dim, MRNG pruning produces nearly the full
  k-NN graph and there is no compression win. Reduce `R` aggressively,
  or fall back to a quantised index.
- **Streaming inserts.** NSG is batch-only by design. For
  insert-heavy workloads, use HNSW or DiskANN's append path; rebuild
  NSG periodically.
- **Memory blow-up at build.** The candidate pool is `O(L_build)` per
  node and we hold all `O(N)` of them transiently; budget at least
  `N × L_build × 8 bytes` of working memory.
- **No persistence in this PoC.** A serde-based snapshot is a tiny
  addition (`(nav, adj, dim)` is all you need) — see "What to improve
  next."

## What to improve next

1. **Parallel build via rayon.** Per-node MRNG is embarrassingly
   parallel; expect 4-8× speedup on M-series cores. (Build time is the
   dominant cost; at N=50k we are at 36s sequential.)
2. **SIMD `l2_sq` for `f32`.** Drop the auto-vectorised inner loop in
   favour of `std::simd` / `wide` for a ~2× search QPS gain on uniform
   data.
3. **Snapshot / persist.** `(nav: u32, adj: Vec<Vec<u32>>, vectors:
   Vec<Vec<f32>>)` round-trips through `bincode` in < 20 lines.
4. **Quantised variant.** Pair with `ruvector-rabitq` for an
   NSG-RaBitQ — 1-bit quantised graph with exact-rerank. Expect 10-20×
   memory reduction at the cost of an extra rerank pass.
5. **Two-pass Vamana-style insert.** Lifts NSG out of "batch only" and
   into "amortised online" without giving up the single-layer
   guarantee.
6. **Real-corpus eval.** SIFT1M / GIST1M / DEEP1M. The synthetic
   uniform numbers here establish correctness and a baseline; SIFT
   numbers anchor the index against the published NSG curves.

## Production crate layout proposal

If we promote `ruvector-nsg` from research to production, the rough
layout would be:

```
crates/ruvector-nsg/
├── src/
│   ├── lib.rs             # AnnIndex impl, public API
│   ├── error.rs
│   ├── knn/
│   │   ├── mod.rs
│   │   ├── nn_descent.rs  # current Step 1
│   │   └── parallel.rs    # rayon variant (NEW)
│   ├── build/
│   │   ├── mod.rs
│   │   ├── mrng.rs        # current Step 3 selection
│   │   └── connect.rs     # DFS + reverse-edge (split out)
│   ├── search.rs          # beam search, SIMD-tuned (NEW)
│   ├── snapshot.rs        # serde + bincode (NEW)
│   └── distance.rs        # l2_sq, ip, cosine (trait-objected)
├── benches/
│   └── nsg_bench.rs
└── tests/
    ├── recall_sift10k.rs  # real-corpus regression (NEW)
    └── determinism.rs
```

Glue with `ruvector-core::index::VectorIndex` is straightforward — NSG's
`(query, k, l_search)` signature already matches; the only adapter
needed is the `add` / `remove` shim (which would just error / mark for
rebuild).

## References

- Cong Fu, Chao Xiang, Changxu Wang, Deng Cai. **"Fast Approximate
  Nearest Neighbor Search With The Navigating Spreading-out Graph."**
  VLDB 2019. <https://arxiv.org/abs/1707.00143>
- Wei Dong, Moses Charikar, Kai Li. **"Efficient k-nearest neighbor
  graph construction for generic similarity measures."** WWW 2011.
- Subramanya et al. **"DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node."** NeurIPS 2019.
- Malkov & Yashunin. **"Efficient and robust approximate nearest
  neighbor search using HNSW."** IEEE TPAMI 2018.
- Cong Fu, Changxu Wang, Deng Cai. **"High Dimensional Similarity
  Search with Satellite System Graph: Efficiency, Scalability, and
  Unindexed Query Compatibility."** 2019.
  <https://arxiv.org/abs/1907.06146>
